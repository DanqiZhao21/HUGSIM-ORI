import sys
import os
sys.path.append(os.getcwd())

import gymnasium
import hugsim_env
from argparse import ArgumentParser
from sim.utils.sim_utils import traj2control, traj_transform_to_global
import pickle
import json
import pickle
from sim.utils.launch_ad import launch, check_alive
from sim.utils.config_loader import load_closed_loop_cfg
from sim.utils.ad_config import resolve_ad_launch_env, resolve_ad_path
from sim.utils.closed_loop_io import read_fifo_with_ad_monitor, write_fifo_with_ad_monitor
from sim.utils.closed_loop_runtime import build_frame_record
from sim.utils.random_seed import seed_everything
from omegaconf import OmegaConf
import open3d as o3d
from sim.utils.score_calculator import hugsim_evaluate
from sim.utils.camera_grid_visualization import build_visual_camera_grid
import numpy as np
from moviepy import ImageSequenceClip

def to_video(observations, output_path):
    frames = []
    for obs in observations:
        frames.append(build_visual_camera_grid(obs))
    clip = ImageSequenceClip(frames, fps=4)
    clip.write_videofile(output_path)


def create_gym_env(cfg, output, *, ad_process=None, pipe_timeout_seconds=300.0):

    env = gymnasium.make('hugsim_env/HUGSim-v0', cfg=cfg, output=output)

    observations_save, infos_save = [], []
    obs, info = env.reset()
    done = False
    cnt = 0
    save_data = {'type': 'closeloop', 'frames': []}

    obs_pipe = os.path.join(output, 'obs_pipe')
    plan_pipe = os.path.join(output, 'plan_pipe')
    if not os.path.exists(obs_pipe):
        os.mkfifo(obs_pipe)
    if not os.path.exists(plan_pipe):
        os.mkfifo(plan_pipe)
    print('Ready for simulation')

    obs, info = None, None
    while not done:

        if obs is None or info is None:
            obs, info = env.reset()
        observations_save.append(obs['rgb'])
        infos_save.append(info)

        print('ego pose', info['ego_pos'])

        write_fifo_with_ad_monitor(
            obs_pipe,
            (obs, info),
            process=ad_process,
            timeout_seconds=pipe_timeout_seconds,
        )
        plan_traj = read_fifo_with_ad_monitor(
            plan_pipe,
            process=ad_process,
            timeout_seconds=pipe_timeout_seconds,
        )

        if plan_traj is not None:
            acc, steer_rate = traj2control(plan_traj, info)

            action = {'acc': acc, 'steer_rate': steer_rate}
            obs, reward, terminated, truncated, info = env.step(action)
            cnt += 1
            done = terminated or truncated or cnt > 400

        else:  # AD Side Crushed
            done = True
            continue

        frame_record = build_frame_record(plan_traj, info)
        if frame_record is not None:
            save_data['frames'].append(frame_record)

    try:
        write_fifo_with_ad_monitor(
            obs_pipe,
            'Done',
            process=ad_process,
            timeout_seconds=min(pipe_timeout_seconds, 5.0),
        )
    except Exception as exc:
        print(f"Failed to notify AD process of completion: {exc}")

    with open(os.path.join(output, 'data.pkl'), 'wb') as wf:
        pickle.dump([save_data], wf)
        
    to_video(observations_save, os.path.join(output, 'video.mp4'))
    with open(os.path.join(output, 'infos.pkl'), 'wb') as wf:
        pickle.dump(infos_save, wf)
    
    ground_xyz = np.asarray(o3d.io.read_point_cloud(os.path.join(output, 'ground.ply')).points)
    scene_xyz = np.asarray(o3d.io.read_point_cloud(os.path.join(output, 'scene.ply')).points)
    results = hugsim_evaluate([save_data], ground_xyz, scene_xyz)
    with open(os.path.join(output, 'eval.json'), 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--scenario_path", type=str, required=True)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--camera_path", type=str, required=True)
    parser.add_argument("--kinematic_path", type=str, required=True)
    parser.add_argument('--ad', default="uniad")
    parser.add_argument('--ad_cuda', default="1")
    args = parser.parse_args()

    seed = seed_everything(os.environ.get("HUGSIM_RANDOM_SEED", "0"))
    print(f"HUGSIM random seed: {seed}")

    cfg, output = load_closed_loop_cfg(
        scenario_path=args.scenario_path,
        base_path=args.base_path,
        camera_path=args.camera_path,
        kinematic_path=args.kinematic_path,
        ad=args.ad,
    )
    os.makedirs(output, exist_ok=True)

    ad_path = resolve_ad_path(cfg, args.ad)
    ad_launch_env = resolve_ad_launch_env(cfg, args.ad)

    process = launch(ad_path, args.ad_cuda, output, extra_env=ad_launch_env)
    pipe_timeout_seconds = float(os.environ.get("HUGSIM_PIPE_TIMEOUT_SECONDS", "300"))
    exit_code = 0
    try:
        create_gym_env(cfg, output, ad_process=process, pipe_timeout_seconds=pipe_timeout_seconds)
        check_alive(process)
    except Exception as e:
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    if exit_code != 0:
        raise SystemExit(exit_code)
    
    # # For debug
    # create_gym_env(cfg, output)
