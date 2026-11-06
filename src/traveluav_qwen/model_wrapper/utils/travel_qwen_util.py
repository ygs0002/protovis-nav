import json
import os
from collections import OrderedDict
from typing import Dict, List, Sequence

import numpy as np
import torch
import transformers
from peft import PeftModel
from transformers import AutoProcessor, AutoTokenizer, CLIPImageProcessor

from ...llm.constants import DEFAULT_HISTORY_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_PVI_TOKEN, DEFAULT_STOP_TOKEN, DEFAULT_WP_TOKEN
from ...llm.geometry import (
    project_this_state2target_state_axis,
    rotation_matrix_from_vector,
    transform_point,
)
from ...llm.preprocess_qwen import preprocess_qwen_visual
from ...llm.qwen_uav_model import Qwen3_VLUAVForConditionalGeneration
from ...llm.rope2d import get_rope_index_3
from ...traj.vis_traj_arch import VisionTrajectoryGenerator


FIXED_DUAL_CAMERA_INDICES = [0, 4]
_EVAL_TEXT_INSTANCE_CACHE: "OrderedDict[tuple, Dict[str, torch.Tensor]]" = OrderedDict()
_EVAL_TEXT_INSTANCE_CACHE_SIZE = max(16, int(os.environ.get("UAV_QWEN_EVAL_TEXT_CACHE_SIZE", "512")))
_EPISODE_INSTRUCTION_CACHE: "OrderedDict[int, str]" = OrderedDict()
_EPISODE_INSTRUCTION_CACHE_SIZE = max(16, int(os.environ.get("UAV_QWEN_EPISODE_INSTR_CACHE_SIZE", "2048")))


def _resolve_model_base(model_path: str, model_base_arg: str = None) -> str:
    if model_base_arg is not None and str(model_base_arg).strip():
        return os.path.expanduser(model_base_arg)

    adapter_cfg = os.path.join(model_path, "adapter_config.json")
    if os.path.isfile(adapter_cfg):
        try:
            with open(adapter_cfg, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            base_from_adapter = cfg.get("base_model_name_or_path", None)
            if isinstance(base_from_adapter, str) and base_from_adapter.strip():
                return os.path.expanduser(base_from_adapter.strip())
        except Exception:
            pass
    return model_path


def _find_existing_file(candidates: Sequence[str], filename: str) -> str:
    for d in candidates:
        if d is None:
            continue
        path = os.path.join(d, filename)
        if os.path.isfile(path):
            return path
    return ""


def _safe_load_state_dict(model, state_dict: Dict[str, torch.Tensor], tag: str):
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k.rsplit(".", 1)[-1] == "pvi_mix_logits":
            continue
        if k in model_state and hasattr(v, "shape") and hasattr(model_state[k], "shape"):
            if tuple(v.shape) != tuple(model_state[k].shape):
                skipped.append((k, tuple(v.shape), tuple(model_state[k].shape)))
                continue
        filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if skipped:
        preview = ", ".join([f"{k}: ckpt{src}->model{dst}" for k, src, dst in skipped[:8]])
        print(
            f"[load_model][WARN] {tag}: skipped {len(skipped)} shape-mismatched keys. "
            f"Examples: {preview}"
        )
    return missing, unexpected


def load_model(args):
    model_path = os.path.expanduser(args.model_path)
    model_base = _resolve_model_base(model_path, args.model_base)

    if os.path.isdir(model_base):
        cfg_path = os.path.join(model_base, "config.json")
        if not os.path.isfile(cfg_path):
            raise ValueError(
                f"model_base='{model_base}' does not contain config.json. "
                "If this is a LoRA checkpoint, pass --model_base as the base Qwen3-VL model path "
                "or ensure adapter_config.json contains base_model_name_or_path."
            )

    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    processor = AutoProcessor.from_pretrained(model_base)
    search_dirs = [model_path, os.path.dirname(model_path.rstrip("/\\"))]
    pvi_token_count = int(getattr(args, "pvi_token_count", 16))
    for directory in search_dirs:
        config_path = os.path.join(directory, "config.json")
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
            pvi_token_count = int(config.get("pvi_token_count", pvi_token_count))
            break
    pvi_token_count = max(1, min(pvi_token_count, 64))

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    attn_impl = "flash_attention_2" if torch.cuda.is_available() else "sdpa"
    model = Qwen3_VLUAVForConditionalGeneration.from_pretrained(
        model_base,
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
        pvi_token_count=pvi_token_count,
    )

    num_new_tokens = tokenizer.add_tokens(
        [DEFAULT_WP_TOKEN, DEFAULT_HISTORY_TOKEN, DEFAULT_PVI_TOKEN, DEFAULT_STOP_TOKEN],
        special_tokens=True,
    )
    if num_new_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
    model.get_special_token_id(
        {
            DEFAULT_WP_TOKEN: tokenizer.convert_tokens_to_ids(DEFAULT_WP_TOKEN),
            DEFAULT_HISTORY_TOKEN: tokenizer.convert_tokens_to_ids(DEFAULT_HISTORY_TOKEN),
            DEFAULT_PVI_TOKEN: tokenizer.convert_tokens_to_ids(DEFAULT_PVI_TOKEN),
            DEFAULT_STOP_TOKEN: tokenizer.convert_tokens_to_ids(DEFAULT_STOP_TOKEN),
        }
    )

    adapter_cfg = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        model = PeftModel.from_pretrained(model, model_path)

    non_lora_path = _find_existing_file(search_dirs, "non_lora_trainables.bin")
    if non_lora_path:
        non_lora_weights = torch.load(non_lora_path, map_location="cpu")
        _safe_load_state_dict(model, non_lora_weights, tag="non_lora_trainables")
        print(f"[load_model] loaded non_lora_trainables from: {non_lora_path}")
    else:
        missing_non_lora_msg = (
            "non_lora_trainables.bin not found in model_path or its parent. "
            "Waypoint/history/orientation heads may stay randomly initialized when loading LoRA checkpoints."
        )
        allow_missing_non_lora = os.environ.get("UAV_QWEN_ALLOW_MISSING_NONLORA", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if os.path.exists(adapter_cfg) and not allow_missing_non_lora:
            raise FileNotFoundError(
                f"[load_model] {missing_non_lora_msg} "
                "Set UAV_QWEN_ALLOW_MISSING_NONLORA=1 to bypass (not recommended)."
            )
        print(f"[load_model][WARN] {missing_non_lora_msg}")

    mm_projector_path = _find_existing_file(search_dirs, "mm_projector.bin")
    if mm_projector_path:
        projector_weights = torch.load(mm_projector_path, map_location="cpu")
        _safe_load_state_dict(model, projector_weights, tag="mm_projector")
        print(f"[load_model] loaded mm_projector from: {mm_projector_path}")
    else:
        print("[load_model][WARN] mm_projector.bin not found in model_path or its parent.")

    model.eval()
    model.requires_grad_(False)

    return tokenizer, model, processor.image_processor


def trajectory_config(vision_tower, image_processor):
    default_vision_config = {
        "model_type": "clip",
        "hidden_act": "silu",
        "hidden_size": 4096,
        "image_aspect_ratio": "square",
        "image_grid_pinpoints": None,
        "image_processor": "./llamavid/processor/clip-patch14-224",
        "initializer_range": 0.02,
        "intermediate_size": 11008,
        "max_position_embeddings": 4096,
        "max_token": 2048,
        "mm_hidden_size": 1408,
        "mm_projector_type": "mlp2x_gelu",
        "mm_use_im_patch_token": False,
        "mm_use_im_start_end": False,
        "mm_vision_select_feature": "patch",
        "mm_vision_select_layer": -2,
        "mm_vision_tower": "./model_zoo/LAVIS/eva_vit_g.pth",
        "torch_dtype": "float16",
    }
    default_vision_config["image_processor"] = image_processor
    default_vision_config["mm_vision_tower"] = vision_tower
    return transformers.CLIPConfig(**default_vision_config)


def load_traj_model(model_args):
    config = trajectory_config(model_args.vision_tower, model_args.image_processor)
    traj_model = VisionTrajectoryGenerator(config)
    traj_weights = torch.load(os.path.join(model_args.traj_model_path, "model_5.pth"), map_location="cpu")
    traj_weights = {k: v.to(torch.bfloat16 if v.is_floating_point() else v.dtype) for k, v in traj_weights.items()}
    traj_model.load_state_dict(traj_weights, strict=False)
    traj_image_processor_path = os.path.expanduser(model_args.image_processor) if model_args.image_processor else None
    if traj_image_processor_path is None:
        raise ValueError("model_args.image_processor is required for trajectory model CLIP preprocessing.")
    traj_image_processor = CLIPImageProcessor.from_pretrained(traj_image_processor_path)
    traj_model.eval()
    traj_model.requires_grad_(False)
    return traj_model, traj_image_processor


def airsim_scene_to_rgb(image):
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise ValueError("An AirSim Scene image must be a uint8 array with shape [H, W, 3].")
    return np.ascontiguousarray(array[..., ::-1]).copy()


def prepare_data_to_traj_model(episodes, waypoints, image_processor, rot_to_targets=None):
    image_list = []
    target_list = []
    for i in range(len(episodes)):
        info = episodes[i]
        target = waypoints[i][0:3]
        rot_to_target = None
        if rot_to_targets is not None and rot_to_targets[i] is not None:
            rot_to_target = rot_to_targets[i]
        rot_0 = info[0]["sensors"]["imu"]["rotation"]
        rot = info[-1]["sensors"]["imu"]["rotation"]
        if rot_to_target is not None:
            target = np.asarray(rot).T @ np.asarray(rot_0) @ np.asarray(rot_to_target) @ np.asarray(target)
        else:
            target = np.asarray(rot).T @ np.asarray(rot_0) @ np.asarray(target)
        image_list.append(airsim_scene_to_rgb(info[-1]["rgb"][0]))
        target_list.append(target)
    images = np.ascontiguousarray(np.stack(image_list, axis=0)).copy()
    image = image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
    target = torch.tensor(np.asarray(target_list), dtype=torch.float32)
    return {"img": image, "target": target}


def transform_to_world(waypoints, episodes):
    waypoints_world = []
    for i in range(len(waypoints)):
        waypoint = waypoints[i]
        ep = episodes[i]
        pos = ep[-1]["sensors"]["state"]["position"]
        rot = ep[-1]["sensors"]["imu"]["rotation"]
        waypoint_world = np.asarray(rot) @ np.asarray(waypoint).T + np.asarray(pos).reshape(3, 1)
        waypoints_world.append(waypoint_world.T)
    return waypoints_world


def _format_round1_human_prompt(instruction: str, image_num: int) -> str:
    image_block = "\n".join([DEFAULT_IMAGE_TOKEN for _ in range(image_num)])
    return (
        f"Current multi-view images:\n{image_block}\n\n"
        f"Instruction:{instruction}\n\n"
        "Please verify if the target has been reached based on the visual information."
    )


def _format_round2_user_prompt(stage: str, delta: str, cur_pos: str) -> str:
    return (
        f"Stage:{stage}\n"
        f"Previous displacement:{delta}\n"
        f"Current position:{cur_pos}\n\n"
        "Please predict the next waypoint in 3D coordinates."
    )


def _extract_instruction(episode_steps: Sequence[Dict]) -> str:
    for step in reversed(list(episode_steps)):
        instruction = step.get("instruction", None)
        if isinstance(instruction, str) and instruction.strip():
            return instruction
    for step in reversed(list(episode_steps)):
        raw_info = step.get("raw_trajectory_info", None)
        if not isinstance(raw_info, dict):
            continue
        instruction = raw_info.get("instruction", None)
        if isinstance(instruction, str) and instruction.strip():
            return instruction
    raise KeyError("No valid `instruction` text found in episode steps.")


def _extract_latest_two_rgb_steps(episode_steps: Sequence[Dict]):
    latest = None
    prev = None
    for step in reversed(episode_steps):
        if "rgb" not in step:
            continue
        if latest is None:
            latest = step
        else:
            prev = step
            break
    if latest is None:
        raise ValueError("No rgb frame found in episode.")
    return latest, prev


def _extract_instruction_cached(episode_steps: Sequence[Dict]) -> str:
    cache_key = int(id(episode_steps))
    cached = _EPISODE_INSTRUCTION_CACHE.get(cache_key, None)

    tail_instruction = None
    if len(episode_steps) > 0:
        tail = episode_steps[-1]
        candidate = tail.get("instruction", None)
        if isinstance(candidate, str) and candidate.strip():
            tail_instruction = candidate.strip()
        elif isinstance(tail.get("raw_trajectory_info", None), dict):
            candidate = tail["raw_trajectory_info"].get("instruction", None)
            if isinstance(candidate, str) and candidate.strip():
                tail_instruction = candidate.strip()

    if tail_instruction is not None:
        if (cached is None) or (tail_instruction != cached):
            _EPISODE_INSTRUCTION_CACHE[cache_key] = tail_instruction
            _EPISODE_INSTRUCTION_CACHE.move_to_end(cache_key)
            while len(_EPISODE_INSTRUCTION_CACHE) > _EPISODE_INSTRUCTION_CACHE_SIZE:
                _EPISODE_INSTRUCTION_CACHE.popitem(last=False)
            return tail_instruction
        _EPISODE_INSTRUCTION_CACHE.move_to_end(cache_key)
        return cached

    if cached is not None:
        _EPISODE_INSTRUCTION_CACHE.move_to_end(cache_key)
        return cached

    instruction = _extract_instruction(episode_steps).strip()
    _EPISODE_INSTRUCTION_CACHE[cache_key] = instruction
    _EPISODE_INSTRUCTION_CACHE.move_to_end(cache_key)
    while len(_EPISODE_INSTRUCTION_CACHE) > _EPISODE_INSTRUCTION_CACHE_SIZE:
        _EPISODE_INSTRUCTION_CACHE.popitem(last=False)
    return instruction


def _build_eval_text_instance(
    source: List[Dict[str, str]],
    tokenizer,
    image_grid_thw: torch.Tensor,
    merged_grid_tokens: Sequence[int],
    merge_size: int,
    stop_token_id: int,
    pvi_token_id: int,
    wp_token_id: int,
) -> Dict[str, torch.Tensor]:
    source_key = tuple((str(x.get("from", x.get("role", ""))), str(x.get("value", x.get("content", "")))) for x in source)
    cache_key = (
        int(id(tokenizer)),
        tuple(int(x) for x in merged_grid_tokens),
        int(merge_size),
        int(stop_token_id) if stop_token_id is not None else -1,
        int(pvi_token_id) if pvi_token_id is not None else -1,
        int(wp_token_id) if wp_token_id is not None else -1,
        source_key,
    )
    cached = _EVAL_TEXT_INSTANCE_CACHE.get(cache_key, None)
    if cached is not None:
        _EVAL_TEXT_INSTANCE_CACHE.move_to_end(cache_key)
        return {k: v.clone() for k, v in cached.items()}

    data_dict = preprocess_qwen_visual([source], tokenizer, grid_thw_image=merged_grid_tokens)
    attention_mask = torch.ones_like(data_dict["input_ids"])
    position_ids, _ = get_rope_index_3(
        spatial_merge_size=merge_size,
        input_ids=data_dict["input_ids"],
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )
    instance = {
        "input_ids": data_dict["input_ids"][0],
        "position_ids": position_ids[:, 0:1, :],
        "attention_mask": attention_mask[0],
    }
    _EVAL_TEXT_INSTANCE_CACHE[cache_key] = {k: v.clone() for k, v in instance.items()}
    _EVAL_TEXT_INSTANCE_CACHE.move_to_end(cache_key)
    while len(_EVAL_TEXT_INSTANCE_CACHE) > _EVAL_TEXT_INSTANCE_CACHE_SIZE:
        _EVAL_TEXT_INSTANCE_CACHE.popitem(last=False)
    return instance


def prepare_data_to_inputs(
    episodes,
    tokenizer,
    image_processor,
    target_point,
    assist_notice=None,
    pvi_token_count: int = 16,
):
    ori_sources = episodes
    latest_rgb_source, prev_rgb_source = _extract_latest_two_rgb_steps(ori_sources)
    images = []
    rgb_views = latest_rgb_source["rgb"]
    for cam_idx in FIXED_DUAL_CAMERA_INDICES:
        if cam_idx >= len(rgb_views):
            raise ValueError(
                f"Episode rgb views has {len(rgb_views)} items, cannot access camera index {cam_idx} "
                "for fixed dual-view inputs (frontcamera + downcamera)."
            )
        images.append(rgb_views[cam_idx])

    rot = np.asarray(ori_sources[0]["sensors"]["imu"]["rotation"])
    pos = np.asarray(ori_sources[0]["sensors"]["state"]["position"])
    deltas = []
    for source in ori_sources:
        if "rgb" not in source:
            continue
        deltas.append(np.asarray(source["sensors"]["state"]["position"]) - pos)
    history_waypoint = np.asarray([(rot.T @ delta) for delta in deltas], dtype=np.float32)

    target_point = np.asarray(rot.T @ (target_point - pos), dtype=np.float32)
    rotation_to_target = rotation_matrix_from_vector(float(target_point[0]), float(target_point[1]))
    history_waypoint = transform_point(history_waypoint, rotation_to_target)

    latest_pos = np.asarray(latest_rgb_source["sensors"]["state"]["position"])
    if prev_rgb_source is not None:
        prev_pos = np.asarray(prev_rgb_source["sensors"]["state"]["position"])
        delta = np.asarray(rot.T @ (latest_pos - prev_pos), dtype=np.float32)
        delta = transform_point(delta, rotation_to_target)
    else:
        delta = np.asarray([0.0, 0.0, -4.5], dtype=np.float32)
    delta = delta / (np.linalg.norm(delta) + 1e-8)
    delta_str = ",".join([str(round(float(x), 1)) for x in delta])
    cur_pos = np.asarray(rot.T @ (latest_pos - pos), dtype=np.float32)
    cur_pos = transform_point(cur_pos, rotation_to_target)
    cur_pos_str = ",".join([str(round(float(x), 1)) for x in cur_pos])

    stage = assist_notice if assist_notice is not None else ("cruise" if len(ori_sources) > 20 else "take off")
    stage = str(stage).strip() if stage is not None else "cruise"
    if stage == "":
        stage = "cruise"
    instruction = _extract_instruction_cached(ori_sources).replace(DEFAULT_IMAGE_TOKEN, "").strip()
    round1_user = _format_round1_human_prompt(instruction=instruction, image_num=len(images))
    round2_user = _format_round2_user_prompt(stage=stage, delta=delta_str, cur_pos=cur_pos_str)
    pvi_token_count = max(1, min(int(pvi_token_count), 64))
    pvi_block = " ".join([DEFAULT_PVI_TOKEN for _ in range(pvi_token_count)])
    phase1_source = [
        {"from": "human", "value": round1_user},
        {"from": "gpt", "value": f"Stop: {DEFAULT_STOP_TOKEN}"},
    ]
    phase2_source = [
        {"from": "human", "value": round1_user},
        {"from": "gpt", "value": f"Stop: {DEFAULT_STOP_TOKEN}"},
        {"from": "human", "value": round2_user},
        {"from": "gpt", "value": f"Next waypoint: {pvi_block} {DEFAULT_WP_TOKEN}"},
    ]

    merge_size = image_processor.merge_size
    safe_images: List[np.ndarray] = []
    for image in images:
        safe_images.append(airsim_scene_to_rgb(image))
    processed = image_processor.preprocess(safe_images, return_tensors="pt")
    pixel_values = processed["pixel_values"]
    image_grid_thw = processed["image_grid_thw"]
    if isinstance(pixel_values, list):
        tensor_list = []
        for value in pixel_values:
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value)
            tensor_list.append(value if value.ndim == 2 else value.reshape(-1, value.shape[-1]))
        pixel_values = torch.cat(tensor_list, dim=0)
    elif not isinstance(pixel_values, torch.Tensor):
        pixel_values = torch.as_tensor(pixel_values)
    if isinstance(image_grid_thw, list):
        image_grid_thw = torch.as_tensor(image_grid_thw)
    elif not isinstance(image_grid_thw, torch.Tensor):
        image_grid_thw = torch.as_tensor(image_grid_thw)
    if image_grid_thw.ndim == 1:
        image_grid_thw = image_grid_thw.unsqueeze(0)
    merged_grid_tokens = [
        int(image_grid_thw[idx].prod().item() // (merge_size**2))
        for idx in range(image_grid_thw.shape[0])
    ]

    pvi_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_PVI_TOKEN)
    stop_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_STOP_TOKEN)
    wp_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_WP_TOKEN)
    phase1_instance = _build_eval_text_instance(
        source=phase1_source,
        tokenizer=tokenizer,
        image_grid_thw=image_grid_thw,
        merged_grid_tokens=merged_grid_tokens,
        merge_size=merge_size,
        stop_token_id=stop_token_id,
        pvi_token_id=pvi_token_id,
        wp_token_id=wp_token_id,
    )
    phase2_instance = _build_eval_text_instance(
        source=phase2_source,
        tokenizer=tokenizer,
        image_grid_thw=image_grid_thw,
        merged_grid_tokens=merged_grid_tokens,
        merge_size=merge_size,
        stop_token_id=stop_token_id,
        pvi_token_id=pvi_token_id,
        wp_token_id=wp_token_id,
    )

    ori_0 = ori_sources[0]["sensors"]["state"]
    ori = ori_sources[-1]["sensors"]["state"]
    target_relative_orientation = project_this_state2target_state_axis(ori, ori_0)["orientation"]

    return {
        "phase1_input_ids": phase1_instance["input_ids"],
        "phase1_position_ids": phase1_instance["position_ids"],
        "phase1_attention_mask": phase1_instance["attention_mask"],
        "phase2_input_ids": phase2_instance["input_ids"],
        "phase2_position_ids": phase2_instance["position_ids"],
        "phase2_attention_mask": phase2_instance["attention_mask"],
        "input_ids": phase2_instance["input_ids"],
        "position_ids": phase2_instance["position_ids"],
        "attention_mask": phase2_instance["attention_mask"],
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "history_waypoint": torch.tensor(history_waypoint, dtype=torch.float32).view(-1),
        "orientation": torch.tensor(target_relative_orientation, dtype=torch.float32).view(-1),
        "prompt": round1_user,
        "stage": stage,
        "assistant_phase2_prefix": f"Next waypoint: {pvi_block} {DEFAULT_WP_TOKEN}",
        "assistant_expected_suffix": f"{pvi_block} {DEFAULT_WP_TOKEN}",
    }, rotation_to_target
