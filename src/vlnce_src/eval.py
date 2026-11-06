import os
from pathlib import Path
import sys

import numpy as np
import torch
import tqdm

sys.path.append(str(Path(str(os.getcwd())).resolve()))
from utils.logger import logger
from utils.utils import *
from src.model_wrapper.travel_llm import TravelModelWrapper
from src.model_wrapper.base_model import BaseModelWrapper
from src.common.param import args, model_args, data_args
from src.vlnce_src.env_uav import AirVLNENV
from src.vlnce_src.assist import Assist
from src.vlnce_src.closeloop_util import EvalBatchState, BatchIterator, setup, CheckPort, initialize_env_eval, is_dist_avail_and_initialized


def eval(model_wrapper: BaseModelWrapper, assist: Assist, eval_env: AirVLNENV, eval_save_dir):
    model_wrapper.eval()
    
    with torch.no_grad():
        dataset = BatchIterator(eval_env)
        end_iter = len(dataset)
        pbar = tqdm.tqdm(total=end_iter)

        while True:
            env_batchs = eval_env.next_minibatch()
            if env_batchs is None:
                break
            batch_state = EvalBatchState(batch_size=eval_env.batch_size, env_batchs=env_batchs, env=eval_env, assist=assist)

            real_bs = getattr(eval_env, "last_batch_real_size", eval_env.batch_size)
            pbar.update(n=real_bs)
            
            assist_notices = batch_state.get_assist_notices()
            inputs, rot_to_targets = model_wrapper.prepare_inputs(
                batch_state.episodes,
                batch_state.target_positions,
                assist_notices,
            )

            for t in range(int(args.maxWaypoints) + 1):
                completed = getattr(eval_env, "last_batch_start_index", int(eval_env.index_data) - int(eval_env.batch_size))
                logger.info('Step: {} \t Completed: {} / {}'.format(t, int(completed), end_iter))

                is_terminate = batch_state.check_batch_termination(t)
                if is_terminate:
                    break

                active_mask = [not bool(batch_state.skips[i]) for i in range(batch_state.batch_size)]
                if not any(active_mask):
                    break
                
                refined_waypoints = model_wrapper.run(
                    inputs=inputs,
                    episodes=batch_state.episodes,
                    rot_to_targets=rot_to_targets,
                    target_positions=batch_state.target_positions,
                )
                wp_logs = []
                for i in range(batch_state.batch_size):
                    if (i < len(batch_state.is_padding) and batch_state.is_padding[i]) or (i < len(batch_state.skips) and batch_state.skips[i]):
                        continue
                    try:
                        cur_pos = np.asarray(
                            batch_state.episodes[i][-1]["sensors"]["state"]["position"],
                            dtype=np.float32,
                        )
                        wp0 = np.asarray(refined_waypoints[i][0], dtype=np.float32)
                        delta = wp0 - cur_pos
                        delta_norm = float(np.linalg.norm(delta))
                        dxyz = [round(float(x), 3) for x in delta.tolist()]
                        wp_logs.append(f"{i}:{dxyz}(n={delta_norm:.3f})")
                    except Exception:
                        wp_logs.append(f"{i}:ERR")
                if len(wp_logs) > 0:
                    wp_msg = f"step={t} waypoint_delta " + " | ".join(wp_logs)
                    logger.info(wp_msg)
                    try:
                        tqdm.tqdm.write(wp_msg)
                    except Exception:
                        print(wp_msg, flush=True)
                eval_env.makeActions(refined_waypoints, active_mask=active_mask)
                outputs = eval_env.get_obs()
                batch_state.update_from_env_output(outputs)
                
                batch_state.predict_dones = model_wrapper.predict_done(batch_state.episodes, batch_state.object_infos)
                batch_state.predict_done_details = getattr(model_wrapper, "last_predict_done_details", [])
                stop_logs = []
                cur_instances = []
                if isinstance(inputs, dict):
                    maybe_instances = inputs.get("instances", [])
                    if isinstance(maybe_instances, list):
                        cur_instances = maybe_instances
                for i, detail in enumerate(batch_state.predict_done_details):
                    if (i < len(batch_state.is_padding) and batch_state.is_padding[i]) or (i < len(batch_state.skips) and batch_state.skips[i]):
                        continue
                    if not isinstance(detail, dict):
                        continue
                    p = detail.get("pred_stop_prob", None)
                    streak = int(detail.get("stop_prob_streak", 0))
                    trig = bool(detail.get("stop_prob_triggered", False))
                    p_txt = "None" if p is None else f"{float(p):.4f}"
                    stage_txt = "None"
                    if i < len(cur_instances) and isinstance(cur_instances[i], dict):
                        st = cur_instances[i].get("stage", None)
                        if st is not None:
                            stage_txt = str(st)
                    stop_logs.append(f"{i}:{p_txt}(s={streak},trig={int(trig)},stage={stage_txt})")
                if len(stop_logs) > 0:
                    stop_msg = f"step={t} stop_prob " + " | ".join(stop_logs)
                    logger.info(stop_msg)
                    try:
                        tqdm.tqdm.write(stop_msg)
                    except Exception:
                        print(stop_msg, flush=True)
                
                batch_state.update_metric()
                
                assist_notices = batch_state.get_assist_notices()
                inputs, _ = model_wrapper.prepare_inputs(batch_state.episodes, batch_state.target_positions, assist_notices)

        try:
            pbar.close()
        except:
            pass


if __name__ == "__main__":
    
    eval_save_path = args.eval_save_path
    eval_json_path = args.eval_json_path
    dataset_path = args.dataset_path
    
    if not os.path.exists(eval_save_path):
        os.makedirs(eval_save_path)
    
    setup()

    assert CheckPort(), 'error port'

    eval_env = initialize_env_eval(dataset_path=dataset_path, save_path=eval_save_path, eval_json_path=eval_json_path)

    if is_dist_avail_and_initialized():
        torch.distributed.destroy_process_group()

    args.DistributedDataParallel = False
    
    model_wrapper = TravelModelWrapper(model_args=model_args, data_args=data_args)
    
    assist = Assist(always_help=args.always_help, use_gt=args.use_gt)

    print("Assist setting: always_help --", args.always_help, "    use_gt --", args.use_gt)
    
    eval(model_wrapper=model_wrapper,
         assist=assist,
         eval_env=eval_env,
         eval_save_dir=eval_save_path)
    
    eval_env.delete_VectorEnvUtil()
