from typing import Dict, List, Sequence

import torch
import transformers

from .constants import DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN


_QWEN_CHAT_TEMPLATE = (
    "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def _ensure_chat_template(tokenizer: transformers.PreTrainedTokenizer):
    if getattr(tokenizer, "_uav_chat_template_ready", False):
        return
    tokenizer.chat_template = _QWEN_CHAT_TEMPLATE
    setattr(tokenizer, "_uav_chat_template_ready", True)


def preprocess_qwen_visual(
    sources: Sequence[Sequence[Dict[str, str]]],
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image: List[int] = None,
    grid_thw_video: List[int] = None,
) -> Dict[str, torch.Tensor]:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."

    grid_thw_image = grid_thw_image or []
    grid_thw_video = grid_thw_video or []

    _ensure_chat_template(tokenizer)

    visual_replicate_index_image = 0
    visual_replicate_index_video = 0
    input_ids = []

    for source in sources:
        if roles.get(source[0]["from"], source[0]["from"]) != roles["human"]:
            source = source[1:]

        input_id = []
        input_id += tokenizer.apply_chat_template([{"role": "system", "content": system_message}])

        for conv in source:
            role = conv.get("role", conv.get("from", "user"))
            content = conv.get("content", conv.get("value", ""))
            role = roles.get(role, role)

            if role == "user":
                if DEFAULT_IMAGE_TOKEN in content:
                    parts = content.split(DEFAULT_IMAGE_TOKEN)
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        repl_cnt = grid_thw_image[visual_replicate_index_image]
                        replacement = "<|vision_start|>" + "<|image_pad|>" * repl_cnt + "<|vision_end|>"
                        new_parts.append(replacement)
                        visual_replicate_index_image += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

                if DEFAULT_VIDEO_TOKEN in content:
                    parts = content.split(DEFAULT_VIDEO_TOKEN)
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        repl_cnt = grid_thw_video[visual_replicate_index_video]
                        replacement = "<|vision_start|>" + "<|video_pad|>" * repl_cnt + "<|vision_end|>"
                        new_parts.append(replacement)
                        visual_replicate_index_video += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            encode_id = tokenizer.apply_chat_template([{"role": role, "content": content}])
            input_id += encode_id
        input_ids.append(input_id)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
    }
