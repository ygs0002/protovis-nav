from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import Qwen3VLForConditionalGeneration
from transformers.utils import ModelOutput

from .constants import DEFAULT_STOP_TOKEN, DEFAULT_WP_TOKEN


@dataclass
class UAVWaypointOutput(ModelOutput):
    predicted_waypoints: Optional[torch.FloatTensor] = None
    predicted_stop_probs: Optional[torch.FloatTensor] = None


def resolve_uav_hidden_size(config) -> int:
    if getattr(config, "hidden_size", None) is not None:
        return int(config.hidden_size)
    text_config = getattr(config, "text_config", None)
    if isinstance(text_config, dict):
        hidden_size = text_config.get("hidden_size")
    else:
        hidden_size = getattr(text_config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("The model config does not specify a text hidden size.")
    return int(hidden_size)


class Qwen3_VLUAVForConditionalGeneration(Qwen3VLForConditionalGeneration):
    def __init__(self, config, pvi_token_count=16):
        super().__init__(config)
        hidden_size = resolve_uav_hidden_size(config)
        reduced_size = max(64, hidden_size // 2)
        self.pvi_token_count = max(1, min(int(pvi_token_count), 64))
        self.waypoint_emb = nn.Embedding(1, hidden_size)
        self.waypoints_fc = nn.Sequential(
            nn.Linear(hidden_size, reduced_size),
            nn.ReLU(),
            nn.Linear(reduced_size, 64),
        )
        self.waypoints_output = nn.Linear(64, 4)
        self.stop_head = nn.Linear(hidden_size, 1)
        self.history_preprocessor = nn.Sequential(
            nn.Linear(3, reduced_size),
            nn.ReLU(),
            nn.Linear(reduced_size, hidden_size),
        )
        self.orientation_preprocessor = nn.Sequential(
            nn.Linear(3, reduced_size),
            nn.ReLU(),
            nn.Linear(reduced_size, hidden_size),
        )
        self.special_token_dict = {}
        self.rope_deltas = None
        self.config.pvi_token_count = self.pvi_token_count

    def get_parameter_or_buffer(self, target: str):
        try:
            return super().get_parameter_or_buffer(target)
        except AttributeError:
            for leaf in ("weight", "bias"):
                marker = f".{leaf}."
                if marker in target:
                    base_target = target.split(marker, 1)[0] + f".{leaf}"
                    try:
                        return super().get_parameter_or_buffer(base_target)
                    except AttributeError:
                        pass
            raise

    def get_special_token_id(self, special_token_dict):
        self.special_token_dict = dict(special_token_dict)

    def _extract_waypoint_states(self, hidden_states, input_ids, attention_mask):
        wp_token_id = self.special_token_dict.get(DEFAULT_WP_TOKEN)
        if wp_token_id is None:
            raise ValueError("Special token id for <wp> is not set.")
        waypoint_states = []
        for row in range(input_ids.shape[0]):
            matches = torch.nonzero(input_ids[row] == wp_token_id, as_tuple=False).flatten()
            if len(matches) == 0:
                if attention_mask is not None:
                    idx = int(attention_mask[row].sum().item()) - 1
                else:
                    idx = input_ids.shape[1] - 1
            else:
                idx = int(matches[-1].item())
            waypoint_states.append(hidden_states[row, idx, :])
        return torch.stack(waypoint_states, dim=0)

    def _extract_stop_states(self, hidden_states, input_ids):
        stop_token_id = self.special_token_dict.get(DEFAULT_STOP_TOKEN)
        if stop_token_id is None:
            raise ValueError("Special token id for <stop> is not set.")
        stop_states = []
        for row in range(input_ids.shape[0]):
            matches = torch.nonzero(input_ids[row] == stop_token_id, as_tuple=False).flatten()
            if len(matches) != 1:
                raise ValueError(f"Sample {row} has {len(matches)} <stop> tokens, expected exactly 1.")
            stop_states.append(hidden_states[row, int(matches[-1].item()), :])
        return torch.stack(stop_states, dim=0)

    def _build_history_context(self, historys, orientations, hidden_dtype, device):
        if historys is None:
            return None
        hist_in_dtype = next(self.history_preprocessor.parameters()).dtype
        ori_in_dtype = next(self.orientation_preprocessor.parameters()).dtype
        history_items = list(historys)
        history_tensors = [h.to(device=device, dtype=hist_in_dtype).view(-1, 3) for h in history_items]
        batch_size = len(history_tensors)
        if batch_size == 0:
            return None
        lengths = torch.tensor([h.shape[0] for h in history_tensors], device=device, dtype=torch.long)
        max_len = int(lengths.max().item())
        if max_len == 0:
            return None
        padded_history = torch.zeros((batch_size, max_len, 3), device=device, dtype=hist_in_dtype)
        for idx, history in enumerate(history_tensors):
            padded_history[idx, :history.shape[0], :] = history
        history_hidden = self.history_preprocessor(padded_history.view(-1, 3))
        history_hidden = history_hidden.view(batch_size, max_len, -1)
        valid_mask = (
            torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        ).unsqueeze(-1).to(dtype=history_hidden.dtype)
        denom = lengths.clamp(min=1).unsqueeze(1).to(dtype=history_hidden.dtype)
        context = ((history_hidden * valid_mask).sum(dim=1) / denom).to(dtype=hidden_dtype)
        if orientations is not None:
            orientations = orientations.to(device=device, dtype=ori_in_dtype)
            context = context + self.orientation_preprocessor(orientations).to(dtype=hidden_dtype)
        return torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        orientations: Optional[torch.FloatTensor] = None,
        historys: Optional[torch.FloatTensor] = None,
        return_waypoints: bool = False,
        return_stop_prob: bool = False,
        return_stop_only: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple, UAVWaypointOutput]:
        if input_ids is None:
            raise ValueError("input_ids is required for waypoint extraction.")
        if rope_deltas is not None and hasattr(self.model, "rope_deltas"):
            self.model.rope_deltas = rope_deltas
        outputs = self.model(
            input_ids=None if inputs_embeds is not None else input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            cache_position=cache_position,
            use_cache=False if use_cache is None else use_cache,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=True,
            **kwargs,
        )
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = getattr(outputs, "last_hidden_state", None)
            if hidden_states is None:
                hidden_states = outputs[0]
            if getattr(outputs, "rope_deltas", None) is not None:
                self.rope_deltas = outputs.rope_deltas
        hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=1e4, neginf=-1e4)

        if return_stop_only:
            stop_states = self._extract_stop_states(hidden_states, input_ids).to(dtype=hidden_states.dtype)
            stop_logits = self.stop_head(stop_states).view(-1)
            stop_logits = torch.nan_to_num(stop_logits, nan=0.0, posinf=20.0, neginf=-20.0)
            return torch.nan_to_num(torch.sigmoid(stop_logits), nan=0.0, posinf=1.0, neginf=0.0)

        waypoint_states = self._extract_waypoint_states(hidden_states, input_ids, attention_mask)
        history_context = self._build_history_context(
            historys, orientations, waypoint_states.dtype, waypoint_states.device
        )
        if history_context is not None:
            waypoint_states = waypoint_states + history_context
        waypoint_states = waypoint_states + self.waypoint_emb.weight[0].to(dtype=waypoint_states.dtype)
        waypoint_states = torch.nan_to_num(waypoint_states, nan=0.0, posinf=1e4, neginf=-1e4)
        predicted_waypoints = self.waypoints_output(self.waypoints_fc(waypoint_states))
        predicted_waypoints = torch.nan_to_num(predicted_waypoints, nan=0.0, posinf=1e4, neginf=-1e4)
        stop_states = self._extract_stop_states(hidden_states, input_ids).to(dtype=predicted_waypoints.dtype)
        stop_logits = self.stop_head(stop_states).view(-1)
        stop_logits = torch.nan_to_num(stop_logits, nan=0.0, posinf=20.0, neginf=-20.0)
        predicted_stop_probs = torch.nan_to_num(torch.sigmoid(stop_logits), nan=0.0, posinf=1.0, neginf=0.0)
        if return_waypoints:
            if return_stop_prob:
                return predicted_waypoints, predicted_stop_probs
            return predicted_waypoints
        if return_stop_prob:
            return predicted_stop_probs
        return UAVWaypointOutput(
            predicted_waypoints=predicted_waypoints,
            predicted_stop_probs=predicted_stop_probs,
        )
