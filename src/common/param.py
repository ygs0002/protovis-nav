import datetime
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import transformers

from utils.CN import CN


@dataclass
class CommonArguments:
    project_prefix: str = field(
        default_factory=lambda: str(Path(str(os.getcwd())).resolve()),
        metadata={"help": "Project root path."},
    )
    run_type: str = field(default="eval", metadata={"help": "Only eval is supported by this runtime entry."})
    name: str = field(default="default", metadata={"help": "Experiment name."})

    maxWaypoints: int = field(default=500, metadata={"help": "Maximum closed-loop eval steps."})
    batchSize: int = field(default=8)

    simulator_tool_port: int = field(default=30000, metadata={"help": "Simulator tool port."})
    DDP_MASTER_PORT: int = field(default=20001, metadata={"help": "DDP master port."})

    gpu_id: int = field(default=3, metadata={"help": "Default simulator GPU id."})
    sim_gpu_ids: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated GPU ids for simulator scenes, e.g. '0,2'. If set, overrides --gpu_id."},
    )
    always_help: bool = field(default=False)
    use_gt: bool = field(default=False)

    dataset_path: Optional[str] = field(default=None)
    eval_json_path: Optional[str] = field(default=None)
    eval_save_path: Optional[str] = field(default=None)
    object_name_json_path: Optional[str] = field(default=None)
    map_spawn_area_json_path: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    pass


@dataclass
class ModelArguments:
    model_path: Optional[str] = field(default="checkpoints/ProtoVis-Nav")
    model_base: Optional[str] = field(default=None)
    traj_model_path: Optional[str] = field(default=None)
    vision_tower: Optional[str] = field(default=None)
    image_processor: Optional[str] = field(default=None)
    pvi_token_count: Optional[int] = field(default=16)


parser = transformers.HfArgumentParser((CommonArguments, ModelArguments, DataArguments))
args, model_args, data_args = parser.parse_args_into_dataclasses()

args.make_dir_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
args.logger_file_name = "{}/workdir/{}/logs/{}_{}.log".format(
    args.project_prefix,
    args.run_type,
    args.name,
    args.make_dir_time,
)

assert args.run_type in ["eval"], "run_type error: ProtoVis runtime supports eval only"

args.machines_info = [
    {
        "MACHINE_IP": "127.0.0.1",
        "SOCKET_PORT": int(args.simulator_tool_port),
        "MAX_SCENE_NUM": 16,
        "open_scenes": [],
    },
]

default_config = CN.clone()
default_config.make_dir_time = args.make_dir_time
default_config.freeze()
