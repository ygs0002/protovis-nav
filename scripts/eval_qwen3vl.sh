#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_ENTRY="${ROOT_DIR}/src/vlnce_src/eval.py"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/src:${PYTHONPATH:-}"
cd "${ROOT_DIR}"

PYTHON_EXE="${PYTHON_EXE:-python3}"
GPU_ID="${GPU_ID:-1}"
SIM_GPU_IDS="${SIM_GPU_IDS:-1,1}"
SIMULATOR_TOOL_PORT="${SIMULATOR_TOOL_PORT:-25000}"
DDP_MASTER_PORT="${DDP_MASTER_PORT:-20001}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_WAYPOINTS="${MAX_WAYPOINTS:-200}"
PVI_TOKEN_COUNT="${PVI_TOKEN_COUNT:-16}"
ALWAYS_HELP="${ALWAYS_HELP:-True}"
USE_GT="${USE_GT:-True}"
STOP_PROB_THRESH="${STOP_PROB_THRESH:-0.75}"
STOP_PROB_CONSECUTIVE="${STOP_PROB_CONSECUTIVE:-1}"

DATASET_PATH="${DATASET_PATH:-${ROOT_DIR}/../TravelUAV_decompressed}"
EVAL_JSON_PATH="${EVAL_JSON_PATH:-${ROOT_DIR}/../TravelUAV_data_json/data/uav_dataset/seen_valset.json}"
EVAL_SAVE_PATH="${EVAL_SAVE_PATH:-${ROOT_DIR}/outputs/seen}"

MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/checkpoints/ProtoVis-Nav}"
MODEL_BASE="${MODEL_BASE:-${ROOT_DIR}/../Qwen3-VL/Qwen3-VL-2B-Instruct}"
TRAJ_MODEL_PATH="${TRAJ_MODEL_PATH:-${ROOT_DIR}/../pretrained_checkpoints/travel-traj-model}"
VISION_TOWER="${VISION_TOWER:-${ROOT_DIR}/../Model/LLaMA-UAV/model_zoo/LAVIS/eva_vit_g.pth}"
IMAGE_PROCESSOR="${IMAGE_PROCESSOR:-${ROOT_DIR}/assets/clip-patch14-224}"

MAP_SPAWN_AREA_JSON_PATH="${MAP_SPAWN_AREA_JSON_PATH:-${ROOT_DIR}/../TravelUAV_data_json/data/meta/map_spawnarea_info.json}"
OBJECT_NAME_JSON_PATH="${OBJECT_NAME_JSON_PATH:-${ROOT_DIR}/../TravelUAV_data_json/data/meta/object_description.json}"

if ! command -v "${PYTHON_EXE}" >/dev/null 2>&1; then
  echo "ERROR: python executable not found: ${PYTHON_EXE}" >&2
  exit 1
fi
if [[ ! -f "${EVAL_ENTRY}" ]]; then
  echo "ERROR: eval entry not found: ${EVAL_ENTRY}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: model checkpoint dir not found: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_BASE}/config.json" ]]; then
  echo "ERROR: MODEL_BASE is not a valid HuggingFace model directory: ${MODEL_BASE}" >&2
  echo "Hint: set MODEL_BASE to Qwen3-VL-2B-Instruct path (contains config.json)." >&2
  exit 1
fi

mkdir -p "${EVAL_SAVE_PATH}"

export UAV_QWEN_STOP_PROB_THRESH="${STOP_PROB_THRESH}"
export UAV_QWEN_STOP_PROB_CONSECUTIVE="${STOP_PROB_CONSECUTIVE}"

"${PYTHON_EXE}" "${EVAL_ENTRY}" \
  --run_type eval \
  --name ProtoVisNav \
  --gpu_id "${GPU_ID}" \
  --sim_gpu_ids "${SIM_GPU_IDS}" \
  --simulator_tool_port "${SIMULATOR_TOOL_PORT}" \
  --DDP_MASTER_PORT "${DDP_MASTER_PORT}" \
  --batchSize "${BATCH_SIZE}" \
  --always_help "${ALWAYS_HELP}" \
  --use_gt "${USE_GT}" \
  --maxWaypoints "${MAX_WAYPOINTS}" \
  --pvi_token_count "${PVI_TOKEN_COUNT}" \
  --dataset_path "${DATASET_PATH}" \
  --eval_save_path "${EVAL_SAVE_PATH}" \
  --model_path "${MODEL_PATH}" \
  --model_base "${MODEL_BASE}" \
  --vision_tower "${VISION_TOWER}" \
  --image_processor "${IMAGE_PROCESSOR}" \
  --traj_model_path "${TRAJ_MODEL_PATH}" \
  --eval_json_path "${EVAL_JSON_PATH}" \
  --map_spawn_area_json_path "${MAP_SPAWN_AREA_JSON_PATH}" \
  --object_name_json_path "${OBJECT_NAME_JSON_PATH}"
