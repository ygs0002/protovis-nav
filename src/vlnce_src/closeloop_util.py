
import json
import random
import shutil

import cv2
import numpy as np
from utils.utils import *
from src.common.param import args
import torch.backends.cudnn as cudnn
from src.vlnce_src.env_uav import AirVLNENV, RGB_FOLDER, DEPTH_FOLDER


def setup(seed_offset=0, manual_init_distributed_mode=False):
    if not manual_init_distributed_mode:
        init_distributed_mode()

    seed = 100 + get_rank() + seed_offset
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = False

def CheckPort():
    pid = FromPortGetPid(int(args.DDP_MASTER_PORT))
    if pid is not None:
        print('DDP_MASTER_PORT ({}) is being used'.format(args.DDP_MASTER_PORT))
        return False

    return True

def initialize_env_eval(dataset_path, save_path, eval_json_path):
    eval_env = AirVLNENV(batch_size=args.batchSize, dataset_path=dataset_path, save_path=save_path, eval_json_path=eval_json_path)
    return eval_env

def save_to_dataset_eval(episodes, path, ori_traj_dir, termination_info=None):
    root_path = os.path.join(path)
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    folder_names = ['log'] + RGB_FOLDER + DEPTH_FOLDER
    for folder_name in folder_names:
        os.makedirs(os.path.join(root_path, folder_name), exist_ok=True)
    print(root_path)
    save_logs(episodes, root_path)
    save_images(episodes, root_path)

    ori_obj = os.path.join(ori_traj_dir, 'object_description.json')
    target_obj = os.path.join(root_path, 'object_description.json')
    shutil.copy2(ori_obj, target_obj)
    ori_info = {'ori_traj_dir': ori_traj_dir}
    if termination_info is not None:
        ori_info['termination_info'] = termination_info
        ori_info['stop_dist_thresh_met'] = bool(termination_info.get('stop_dist_thresh_met', False))
        ori_info['stop_prob_thresh_met'] = bool(termination_info.get('stop_prob_thresh_met', False))
    with open(os.path.join(path, 'ori_info.json'), 'w') as f:
        json.dump(ori_info, f)
    if termination_info is not None:
        with open(os.path.join(path, 'termination_info.json'), 'w') as f:
            json.dump(termination_info, f)

def save_logs(episodes, trajectory_dir):
    save_dir = os.path.join(trajectory_dir, 'log')
    for idx, episode in enumerate(episodes):
        info = {'frame': idx, 'sensors': episode['sensors']}
        with open(os.path.join(save_dir, str(idx).zfill(6) + '.json'), 'w') as f:
            json.dump(info, f)

def save_images(episodes, trajectory_dir):
    for idx, episode in enumerate(episodes):
        if 'rgb' in episode:
            for cid, camera_name in enumerate(RGB_FOLDER):
                image = episode['rgb'][cid]
                cv2.imwrite(os.path.join(trajectory_dir, camera_name, str(idx).zfill(6) + '.png'), image)
        if 'depth' in episode:
            for cid, camera_name in enumerate(DEPTH_FOLDER):
                image = episode['depth'][cid]
                cv2.imwrite(os.path.join(trajectory_dir, camera_name, str(idx).zfill(6) + '.png'), image)

def target_distance_increasing_for_10frames(lst):
    if len(lst) < 20:
        return False
    sublist = lst[-20:]
    for i in range(1, len(sublist)):
        if sublist[i] < sublist[i - 1]:
            return False
    return True

class BatchIterator:
    def __init__(self, env: AirVLNENV):
        self.env = env
    
    def __len__(self):
        return len(self.env.data)
    
    def __next__(self):
        batch = self.env.next_minibatch()
        if batch is None:
            raise StopIteration
        return batch
    
    def __iter__(self):
        batch = self.env.next_minibatch()
        if batch is None:
            raise StopIteration
        return batch

class EvalBatchState:
    def __init__(self, batch_size, env_batchs, env, assist):
        self.batch_size = batch_size
        self.eval_env = env
        self.assist = assist
        self.episodes = [[] for _ in range(batch_size)]
        self.is_padding = [bool(b.get('__pad__', False)) for b in env_batchs]
        self.target_positions = [b['object_position'] for b in env_batchs]
        self.object_infos = [self._get_object_info(b) for b in env_batchs]
        self.trajs = [b['trajectory'] for b in env_batchs]
        self.ori_data_dirs = [b['trajectory_dir'] for b in env_batchs]
        self.dones = [False] * batch_size
        self.predict_dones = [False] * batch_size
        self.predict_done_details = [{} for _ in range(batch_size)]
        self.collisions = [False] * batch_size
        self.success = [False] * batch_size
        self.oracle_success = [False] * batch_size
        self.early_end = [False] * batch_size
        self.skips = self.is_padding.copy()
        self.distance_to_ends = [[] for _ in range(batch_size)]
        self.envs_to_pause = []
        
        self._initialize_batch_data()

    def _get_object_info(self, batch):
        object_desc_dict = self._load_object_description()
        return object_desc_dict.get(batch['object']['asset_name'].replace("AA", ""))

    def _load_object_description(self):
        with open(args.object_name_json_path, 'r') as f:
            return {item['object_name']: item['object_desc'] for item in json.load(f)}

    def _initialize_batch_data(self):
        outputs = self.eval_env.reset()
        observations, self.dones, self.collisions, self.oracle_success = [list(x) for x in zip(*outputs)]
        
        for i in range(self.batch_size):
            if i in self.envs_to_pause:
                continue
            self.episodes[i].append(observations[i][-1])
            self.distance_to_ends[i].append(self._calculate_distance(observations[i][-1], self.target_positions[i]))

    def _calculate_distance(self, observation, target_position):
        return np.linalg.norm(np.array(observation['sensors']['state']['position']) - np.array(target_position))

    def update_from_env_output(self, outputs):
        observations, self.dones, self.collisions, self.oracle_success = [list(x) for x in zip(*outputs)]
        self.collisions, self.dones = self.assist.check_collision_by_depth(self.episodes, observations, self.collisions, self.dones)
        
        for i in range(self.batch_size):
            if i in self.envs_to_pause:
                continue
            for j in range(len(observations[i])):
                self.episodes[i].append(observations[i][j])
            self.distance_to_ends[i].append(self._calculate_distance(observations[i][-1], self.target_positions[i]))
            if target_distance_increasing_for_10frames(self.distance_to_ends[i]):
                self.collisions[i] = True
                self.dones[i] = True

    def get_assist_notices(self):
        return self.assist.get_assist_notice(self.episodes, self.trajs, self.object_infos, self.target_positions)

    def update_metric(self):
        for i in range(self.batch_size):
            if self.dones[i]:
                continue
            if self.predict_dones[i] and not self.skips[i]:
                cur_dist = float(self.distance_to_ends[i][-1]) if len(self.distance_to_ends[i]) > 0 else float("inf")
                min_dist_hist = (
                    float(np.min(np.asarray(self.distance_to_ends[i], dtype=np.float32)))
                    if len(self.distance_to_ends[i]) > 0
                    else cur_dist
                )

                success_radius = 20.0
                self.success[i] = bool(cur_dist <= success_radius)

                ever_enter_success_radius = bool(min_dist_hist <= success_radius)
                oracle_hit = bool(self.oracle_success[i] or ever_enter_success_radius)
                self.oracle_success[i] = bool((not self.success[i]) and oracle_hit)

                self.dones[i] = True
                    
    def check_batch_termination(self, t):
        for i in range(self.batch_size):
            if t == args.maxWaypoints:
                self.dones[i] = True
            if self.dones[i] and not self.skips[i]:
                predict_detail = {}
                if isinstance(self.predict_done_details, list) and i < len(self.predict_done_details):
                    if isinstance(self.predict_done_details[i], dict):
                        predict_detail = self.predict_done_details[i]
                stop_prob_thresh_met = bool(predict_detail.get("stop_prob_triggered", False))
                done_reasons = []
                if self.collisions[i]:
                    done_reasons.append("collision")
                if self.predict_dones[i]:
                    done_reasons.append("predict_done")
                if stop_prob_thresh_met:
                    done_reasons.append("stop_prob_thresh_met")
                if self.success[i]:
                    done_reasons.append("success")
                if self.oracle_success[i]:
                    done_reasons.append("oracle_success")
                if t == args.maxWaypoints:
                    done_reasons.append("max_waypoints")
                if not done_reasons:
                    done_reasons.append("done_flag")

                self.envs_to_pause.append(i)
                prex = ''
                if self.success[i]:
                    prex = 'success_'
                    print(i, " has succeed!")
                elif self.oracle_success[i]:
                    prex = "oracle_"
                    print(i, " has oracle succeed!")
                new_traj_name = prex +  self.ori_data_dirs[i].split('/')[-1]
                new_traj_dir = os.path.join(args.eval_save_path, new_traj_name)
                termination_info = {
                    "done_reasons": done_reasons,
                    "stop_dist_thresh_met": False,
                    "stop_prob_thresh_met": stop_prob_thresh_met,
                    "stop_mode": predict_detail.get("stop_mode"),
                    "predict_done": bool(self.predict_dones[i]),
                    "predict_done_by_stop": predict_detail.get("predict_done_by_stop"),
                    "pred_stop_prob": predict_detail.get("pred_stop_prob"),
                    "stop_prob_threshold": predict_detail.get("stop_prob_threshold"),
                    "stop_prob_consecutive": predict_detail.get("stop_prob_consecutive"),
                    "stop_prob_streak": predict_detail.get("stop_prob_streak"),
                }
                save_to_dataset_eval(
                    self.episodes[i],
                    new_traj_dir,
                    self.ori_data_dirs[i],
                    termination_info=termination_info,
                )
                self.skips[i] = True
                print(
                    i,
                    " has finished! reason:",
                    ",".join(done_reasons),
                    "| stop_prob_thresh_met=",
                    stop_prob_thresh_met,
                    "| pred_stop_prob=",
                    predict_detail.get("pred_stop_prob"),
                    "| stop_prob_threshold=",
                    predict_detail.get("stop_prob_threshold"),
                    "| stop_prob_streak=",
                    predict_detail.get("stop_prob_streak"),
                    "/",
                    predict_detail.get("stop_prob_consecutive"),
                )
        return np.array(self.skips).all()
