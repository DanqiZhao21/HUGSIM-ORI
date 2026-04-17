
#每次跑之前先清理
ps -eo pid,etime,cmd --sort=-rss | rg 'closed_loop.py|e2e.py|ltf_e2e.py|run_sparsedrive_bridge.py|sparsedrive_e2e.py'

pkill -f 'tools/closeloop/e2e.py'
pkill -f 'ltf_e2e.py'
pkill -f 'run_sparsedrive_bridge.py'
pkill -f 'sparsedrive_e2e.py'

#LFT

CUDA_VISIBLE_DEVICES=0 pixi run python /root/clone/HUGSIM-ORI/closed_loop.py   --scenario_path /root/clone/HUGSIM-ORI/configs/scenarios/nuscenes/scene-0051-easy-00.yaml   --base_path /root/clone/HUGSIM-ORI/configs/sim/nuscenes_base.yaml   --camera_path /root/clone/HUGSIM-ORI/configs/sim/nuscenes_camera.yaml   --kinematic_path /root/clone/HUGSIM-ORI/configs/sim/kinematic.yaml   --ad sparsedrive-v2   --ad_cuda 1


CUDA_VISIBLE_DEVICES=0 pixi run python /root/clone/HUGSIM-ORI/closed_loop.py \
  --scenario_path /root/clone/HUGSIM-ORI/configs/scenarios/kitti360/scene-250_450-easy-00.yaml \
  --base_path /root/clone/HUGSIM-ORI/configs/sim/kitti360_base.yaml \
  --camera_path /root/clone/HUGSIM-ORI/configs/sim/kitti360_camera.yaml \
  --kinematic_path /root/clone/HUGSIM-ORI/configs/sim/kinematic.yaml \
  --ad ltf \
  --ad_cuda 1

#sparsedrive
