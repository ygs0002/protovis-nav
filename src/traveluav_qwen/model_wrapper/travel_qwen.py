import numpy as np
import torch
import os
from typing import Dict, List, Sequence
from ..llm.constants import DEFAULT_PVI_TOKEN, DEFAULT_WP_TOKEN

from .base_model import BaseModelWrapper
from .utils.travel_qwen_util import (
    load_model,
    load_traj_model,
    prepare_data_to_inputs,
    prepare_data_to_traj_model,
    transform_to_world,
)


class TravelQwenWrapper(BaseModelWrapper):
    def __init__(self, model_args, data_args=None):
        self.tokenizer, self.model, self.image_processor = load_model(model_args)
        self.traj_model, self.traj_image_processor = load_traj_model(model_args)
        self.model_args = model_args
        self.data_args = data_args
        self.last_pred_stop_probs = None
        self._last_assist_notices = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.traj_model.to(device=self.device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)

    def _get_model_pvi_token_count(self) -> int:
        model_cfg = getattr(self.model, "config", None)
        pvi_token_count = getattr(model_cfg, "pvi_token_count", None)
        if pvi_token_count is None:
            raise ValueError(
                "model.config.pvi_token_count is missing in eval. "
                "Please load a checkpoint/config that records PVI token count."
            )
        return max(1, min(int(pvi_token_count), 64))

    def _assert_instance_pvi_suffix(self, instance: Dict) -> None:
        expected_pvi_count = self._get_model_pvi_token_count()
        suffix = str(instance.get("assistant_expected_suffix", ""))
        actual_pvi_count = sum(1 for token in suffix.replace("\n", " ").split() if token == DEFAULT_PVI_TOKEN)
        if actual_pvi_count != expected_pvi_count:
            raise ValueError(
                "assistant_expected_suffix PVI token mismatch in eval: "
                f"suffix_pvi_count={actual_pvi_count}, model.config.pvi_token_count={expected_pvi_count}, "
                f"suffix='{suffix}'"
            )
        phase2_input_ids = instance.get("phase2_input_ids")
        if torch.is_tensor(phase2_input_ids):
            pvi_token_id = int(self.tokenizer.convert_tokens_to_ids(DEFAULT_PVI_TOKEN))
            wp_token_id = int(self.tokenizer.convert_tokens_to_ids(DEFAULT_WP_TOKEN))
            phase2_pvi_count = int((phase2_input_ids == pvi_token_id).sum().item())
            phase2_wp_count = int((phase2_input_ids == wp_token_id).sum().item())
            if phase2_pvi_count < expected_pvi_count or phase2_wp_count < 1:
                raise ValueError(
                    "Phase2 eval template is missing required PVI/WP tokens: "
                    f"phase2_pvi_count={phase2_pvi_count}, expected_pvi_count={expected_pvi_count}, "
                    f"phase2_wp_count={phase2_wp_count}."
                )

    def prepare_inputs(self, episodes, target_positions, assist_notices=None):
        instances = []
        rot_to_targets = []
        pvi_token_count = self._get_model_pvi_token_count()
        for i in range(len(episodes)):
            instance, rot_to_target = prepare_data_to_inputs(
                episodes=episodes[i],
                tokenizer=self.tokenizer,
                image_processor=self.image_processor,
                target_point=target_positions[i],
                assist_notice=assist_notices[i] if assist_notices is not None else None,
                pvi_token_count=pvi_token_count,
            )
            self._assert_instance_pvi_suffix(instance)
            instances.append(instance)
            rot_to_targets.append(rot_to_target)

        self._last_assist_notices = list(assist_notices) if assist_notices is not None else None
        inputs = {"instances": instances}
        return inputs, rot_to_targets

    def _build_phase_batch(self, instances: List[Dict], phase: str, indices: Sequence[int]) -> Dict[str, torch.Tensor]:
        selected = [instances[int(i)] for i in indices]
        if len(selected) == 0:
            raise ValueError("No selected instances for phase batch.")

        key_input_ids = f"{phase}_input_ids"
        key_position_ids = f"{phase}_position_ids"
        key_attention_mask = f"{phase}_attention_mask"
        input_ids = [x[key_input_ids] for x in selected]
        position_ids = [x[key_position_ids] for x in selected]
        attention_masks = [x[key_attention_mask] for x in selected]

        max_len = max(x.shape[0] for x in input_ids)
        bs = len(input_ids)
        padded_input_ids = torch.full(
            (bs, max_len),
            int(self.tokenizer.pad_token_id),
            dtype=input_ids[0].dtype,
        )
        padded_attention = torch.zeros((bs, max_len), dtype=attention_masks[0].dtype)
        padded_positions = []
        for i, (ids, attn, pos) in enumerate(zip(input_ids, attention_masks, position_ids)):
            cur_len = int(ids.shape[0])
            padded_input_ids[i, max_len - cur_len :] = ids
            padded_attention[i, max_len - cur_len :] = attn
            padded_positions.append(torch.nn.functional.pad(pos, (max_len - pos.shape[2], 0), "constant", 1))
        input_ids = padded_input_ids[:, -self.tokenizer.model_max_length :]
        attention_mask = padded_attention[:, -self.tokenizer.model_max_length :]
        position_ids = torch.cat(padded_positions, dim=1)
        position_ids = position_ids[:, :, -self.tokenizer.model_max_length :]

        pixel_values = torch.cat([x["pixel_values"] for x in selected], dim=0)
        image_grid_thw = torch.cat([x["image_grid_thw"] for x in selected], dim=0)
        historys = [
            x["history_waypoint"].to(
                device=self.device, dtype=torch.float32 if self.device.type == "cpu" else torch.bfloat16
            )
            for x in selected
        ]
        orientations = torch.stack([x["orientation"] for x in selected], dim=0).to(device=self.device)

        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "position_ids": position_ids.to(self.device),
            "pixel_values": pixel_values.to(self.device),
            "image_grid_thw": image_grid_thw.to(self.device),
            "historys": historys,
            "orientations": orientations,
        }

    def _phase1_trigger_mask(self, pred_stop_probs: np.ndarray, episodes) -> List[bool]:
        bs = int(len(pred_stop_probs))
        threshold = float(getattr(self, "stop_prob_threshold", 0.5))
        consecutive = max(1, int(getattr(self, "stop_prob_consecutive", 2)))
        streaks = getattr(self, "_stop_prob_streaks", None)
        if not isinstance(streaks, list) or len(streaks) != bs:
            streaks = [0 for _ in range(bs)]
        triggered = []
        for i in range(bs):
            prev = 0 if len(episodes[i]) <= 1 else int(streaks[i])
            p = float(pred_stop_probs[i])
            cond = bool((p == p) and np.isfinite(p) and p >= threshold)
            nxt = prev + 1 if cond else 0
            triggered.append(bool(nxt >= consecutive))
        return triggered

    def _phase2_generate_and_decode(self, phase2_inputs: Dict[str, torch.Tensor]) -> np.ndarray:
        decode_out = self.model(
            input_ids=phase2_inputs["input_ids"],
            attention_mask=phase2_inputs["attention_mask"],
            position_ids=phase2_inputs["position_ids"],
            pixel_values=phase2_inputs["pixel_values"],
            image_grid_thw=phase2_inputs["image_grid_thw"],
            historys=phase2_inputs["historys"],
            orientations=phase2_inputs["orientations"],
            return_waypoints=True,
            return_stop_prob=False,
            use_cache=False,
        )
        if isinstance(decode_out, tuple):
            pred_waypoints = decode_out[0]
        elif hasattr(decode_out, "predicted_waypoints"):
            pred_waypoints = decode_out.predicted_waypoints
        else:
            pred_waypoints = decode_out
        if not torch.is_tensor(pred_waypoints):
            raise TypeError(f"Unexpected LLM output type for waypoints: {type(pred_waypoints)}")
        if pred_waypoints.ndim == 1:
            pred_waypoints = pred_waypoints.unsqueeze(0)
        if pred_waypoints.ndim != 2 or pred_waypoints.shape[-1] != 4:
            raise ValueError(
                f"Unexpected waypoint output shape from LLM: {tuple(pred_waypoints.shape)}. Expect [B, 4]."
            )
        return pred_waypoints.detach().cpu().to(dtype=torch.float32).numpy()

    def run_llm_model(self, inputs, episodes):
        instances = list(inputs["instances"])
        all_indices = list(range(len(instances)))
        phase1_inputs = self._build_phase_batch(instances, phase="phase1", indices=all_indices)
        phase1_out = self.model(
            input_ids=phase1_inputs["input_ids"],
            attention_mask=phase1_inputs["attention_mask"],
            position_ids=phase1_inputs["position_ids"],
            pixel_values=phase1_inputs["pixel_values"],
            image_grid_thw=phase1_inputs["image_grid_thw"],
            return_waypoints=False,
            return_stop_prob=True,
            return_stop_only=True,
            use_cache=False,
        )
        if not torch.is_tensor(phase1_out):
            raise TypeError(f"Unexpected stop-only output type: {type(phase1_out)}")
        pred_stop_probs = phase1_out.view(-1).detach().cpu().to(dtype=torch.float32).numpy()
        self.last_pred_stop_probs = pred_stop_probs
        stop_triggered = self._phase1_trigger_mask(pred_stop_probs=pred_stop_probs, episodes=episodes)

        active_indices = [idx for idx in all_indices if not stop_triggered[idx]]
        all_waypoints_llm = np.zeros((len(instances), 4), dtype=np.float32)
        if len(active_indices) > 0:
            phase2_inputs = self._build_phase_batch(instances, phase="phase2", indices=active_indices)
            active_waypoints = self._phase2_generate_and_decode(phase2_inputs)
            for j, idx in enumerate(active_indices):
                all_waypoints_llm[idx] = active_waypoints[j]

        waypoints_llm_new = []
        force_nonneg_norm = os.environ.get("UAV_QWEN_FORCE_NONNEG_NORM", "0").strip().lower() in {"1", "true", "yes", "on"}
        debug_waypoint = os.environ.get("UAV_QWEN_DEBUG_WAYPOINT", "0").strip().lower() in {"1", "true", "yes", "on"}
        llm_sign = self._parse_axis_sign_from_env("UAV_QWEN_LLM_SIGN", default=(1.0, 1.0, 1.0))
        for waypoint in all_waypoints_llm:
            wp_norm = float(waypoint[3])
            if force_nonneg_norm:
                wp_norm = abs(wp_norm)
            waypoint_new = waypoint[:3] / (1e-6 + np.linalg.norm(waypoint[:3])) * wp_norm
            waypoint_new = waypoint_new * llm_sign
            waypoints_llm_new.append(waypoint_new)
        if debug_waypoint and len(all_waypoints_llm) > 0:
            first_raw = all_waypoints_llm[0]
            first_vec = waypoints_llm_new[0]
            print(
                "[UAV_QWEN_DEBUG_WAYPOINT] raw_first=",
                np.round(first_raw, 6).tolist(),
                "norm_used=",
                round(float(abs(first_raw[3]) if force_nonneg_norm else first_raw[3]), 6),
                "vec_first=",
                np.round(first_vec, 6).tolist(),
                "force_nonneg_norm=",
                force_nonneg_norm,
                "llm_sign=",
                np.round(llm_sign, 4).tolist(),
                "pred_stop_prob_first=",
                None if self.last_pred_stop_probs is None else round(float(self.last_pred_stop_probs[0]), 6),
            )
        return np.asarray(waypoints_llm_new, dtype=np.float32), stop_triggered

    @staticmethod
    def _parse_axis_sign_from_env(env_name: str, default=(1.0, 1.0, 1.0)):
        raw = os.environ.get(env_name, "").strip()
        if not raw:
            return np.asarray(default, dtype=np.float32)
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 3:
            raise ValueError(f"{env_name} must be three comma-separated values, e.g. '1,1,1' or '-1,-1,-1'.")
        vals = []
        for p in parts:
            v = float(p)
            if abs(v) < 1e-8:
                raise ValueError(f"{env_name} contains zero value '{p}', each axis scale should be non-zero.")
            vals.append(v)
        return np.asarray(vals, dtype=np.float32)

    @staticmethod
    def _convert_target_frame_to_current_frame(episodes, waypoints_target_frame, rot_to_targets):
        targets_current_frame = []
        for i in range(len(episodes)):
            info = episodes[i]
            target = np.asarray(waypoints_target_frame[i][0:3], dtype=np.float32)
            rot_to_target = None
            if rot_to_targets is not None and rot_to_targets[i] is not None:
                rot_to_target = np.asarray(rot_to_targets[i], dtype=np.float32)
            rot_0 = np.asarray(info[0]["sensors"]["imu"]["rotation"], dtype=np.float32)
            rot = np.asarray(info[-1]["sensors"]["imu"]["rotation"], dtype=np.float32)
            if rot_to_target is not None:
                target_current = rot.T @ rot_0 @ rot_to_target @ target
            else:
                target_current = rot.T @ rot_0 @ target
            targets_current_frame.append(target_current)
        return np.asarray(targets_current_frame, dtype=np.float32)

    def _build_world_path_from_current_vectors(self, episodes, vecs_current_frame, num_points: int = 5):
        paths = []
        for i in range(len(episodes)):
            ep = episodes[i]
            pos = np.asarray(ep[-1]["sensors"]["state"]["position"], dtype=np.float32)
            rot = np.asarray(ep[-1]["sensors"]["imu"]["rotation"], dtype=np.float32)
            world_step = rot @ np.asarray(vecs_current_frame[i], dtype=np.float32)
            path = np.stack([pos + world_step * ((k + 1) / float(num_points)) for k in range(num_points)], axis=0)
            paths.append(path)
        return paths

    def run_traj_model(self, episodes, waypoints_llm_new, rot_to_targets, target_positions=None):
        debug_chain = os.environ.get("UAV_QWEN_DEBUG_CHAIN", "0").strip().lower() in {"1", "true", "yes", "on"}
        bypass_traj = os.environ.get("UAV_QWEN_BYPASS_TRAJ", "0").strip().lower() in {"1", "true", "yes", "on"}
        traj_sign = self._parse_axis_sign_from_env("UAV_QWEN_TRAJ_SIGN", default=(1.0, 1.0, 1.0))

        llm_targets_current = self._convert_target_frame_to_current_frame(episodes, waypoints_llm_new, rot_to_targets)
        if bypass_traj:
            return self._build_world_path_from_current_vectors(episodes, llm_targets_current * traj_sign, num_points=5)

        inputs = prepare_data_to_traj_model(episodes, waypoints_llm_new, self.traj_image_processor, rot_to_targets)
        waypoints_traj = self.traj_model(inputs)
        refined_waypoints = waypoints_traj.detach().cpu().to(dtype=torch.float32).numpy()
        refined_waypoints = refined_waypoints * traj_sign.reshape(1, 1, 3)
        if debug_chain and len(refined_waypoints) > 0:
            for i in range(min(len(refined_waypoints), 3)):
                ep = episodes[i]
                cur_pos = np.asarray(ep[-1]["sensors"]["state"]["position"], dtype=np.float32)
                cur_rot = np.asarray(ep[-1]["sensors"]["imu"]["rotation"], dtype=np.float32)
                llm_world = cur_rot @ llm_targets_current[i]
                traj_local = np.asarray(refined_waypoints[i][0], dtype=np.float32)
                traj_world = cur_rot @ traj_local

                msg = (
                    f"[UAV_QWEN_DEBUG_CHAIN] i={i} "
                    f"llm_current={np.round(llm_targets_current[i], 4).tolist()} "
                    f"llm_world={np.round(llm_world, 4).tolist()} "
                    f"traj_local0={np.round(traj_local, 4).tolist()} "
                    f"traj_world0={np.round(traj_world, 4).tolist()}"
                )
                if target_positions is not None:
                    target_world = np.asarray(target_positions[i], dtype=np.float32)
                    goal_world = target_world - cur_pos
                    goal_norm = float(np.linalg.norm(goal_world) + 1e-8)
                    llm_cos = float(np.dot(llm_world, goal_world) / ((np.linalg.norm(llm_world) + 1e-8) * goal_norm))
                    traj_cos = float(np.dot(traj_world, goal_world) / ((np.linalg.norm(traj_world) + 1e-8) * goal_norm))
                    msg += (
                        f" goal_world={np.round(goal_world, 4).tolist()} llm_goal_cos={llm_cos:.4f} "
                        f"traj_goal_cos={traj_cos:.4f} traj_sign={np.round(traj_sign, 4).tolist()}"
                    )
                print(msg)
        return transform_to_world(refined_waypoints, episodes)

    def eval(self):
        self.model.eval()
        self.traj_model.eval()

    @torch.inference_mode()
    def run(self, inputs, episodes, rot_to_targets, target_positions=None):
        waypoints_llm_new, stop_triggered = self.run_llm_model(inputs, episodes)
        refined_waypoints = self.run_traj_model(
            episodes,
            waypoints_llm_new,
            rot_to_targets,
            target_positions=target_positions,
        )
        stop_indices = [i for i, v in enumerate(stop_triggered) if bool(v)]
        if len(stop_indices) > 0:
            zero_vecs = np.zeros((len(stop_indices), 3), dtype=np.float32)
            selected_eps = [episodes[i] for i in stop_indices]
            stationary = self._build_world_path_from_current_vectors(selected_eps, zero_vecs, num_points=5)
            for local_i, global_i in enumerate(stop_indices):
                refined_waypoints[global_i] = stationary[local_i]
        return refined_waypoints
