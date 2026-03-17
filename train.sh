#python legged_gym/scripts/train.py --task=go2_stair --experiment_name go2_terrain_0_target_exp --run_name go2_terrain_0_target_exp_0 --num_envs 4096 --seed 42 --max_iterations 10000 --sim_device cuda:2 --rl_device cuda:1 --headless

#nohup python legged_gym/scripts/train.py --task=go2_stair --experiment_name go2_terrain_0_exp --run_name go2_terrain_0_exp_0 --sim_device cuda:0 --rl_device cuda:0 --num_envs 4096 --seed 42 --max_iterations 10000 --headless >go2_terrain_0_exp_0.out 2>&1 &
#nohup python legged_gym/scripts/train.py --task=go2_stair --experiment_name go2_terrain_2_exp --run_name go2_terrain_2_exp_0 --sim_device cuda:1 --rl_device cuda:1 --num_envs 4096 --seed 42 --max_iterations 10000 --headless >go2_terrain_2_exp_0.out 2>&1 &
#nohup python legged_gym/scripts/train.py --task=go2_stair --experiment_name go2_terrain_4_exp --run_name go2_terrain_4_exp_0 --sim_device cuda:2 --rl_device cuda:2 --num_envs 4096 --seed 42 --max_iterations 10000 --headless >go2_terrain_4_exp_0.out 2>&1 &
#nohup python legged_gym/scripts/train.py --task=go2_stair --experiment_name go2_terrain_2_exp --run_name go2_terrain_2_exp_2 --sim_device cuda:0 --rl_device cuda:0 --num_envs 4096 --seed 42 --max_iterations 10000 --headless >go2_terrain_2_exp_2.out 2>&1 &
nohup python legged_gym/scripts/train.py --task=go2_terrain_navigation --experiment_name go2_terrain_navigation_2_exp --run_name 5 --sim_device cuda:3 --rl_device cuda:3 --num_envs 4096 --seed 42 --max_iterations 10000 --headless >go2_terrain_2_exp_2.out 2>&1 &

#python legged_gym/scripts/train.py --task=go2_stair --experiment_name debug --run_name debug --sim_device cuda:0 --rl_device cuda:0 --num_envs 4096 --seed 42 --max_iterations 10000
python legged_gym/scripts/train.py --task=go2_terrain_navigation --experiment_name debug --run_name debug --sim_device cuda:0 --rl_device cuda:0 --num_envs 4096 --seed 42 --max_iterations 10000

nohup python legged_gym/scripts/train.py --task=g1_terrain --experiment_name g1_terrain_exp --run_name g1_terrain_7_20cm --sim_device cuda:2 --rl_device cuda:2 --num_envs 4096 --seed 42 --max_iterations 10000 --headless >g1_terrain_2.out 2>&1 &
python legged_gym/scripts/train.py --task=g1_terrain --experiment_name debug --run_name debug --sim_device cuda:2 --rl_device cuda:2 --num_envs 64 --seed 42 --max_iterations 10000
#python legged_gym/scripts/train.py --task=g1_terrain --experiment_name debug --run_name debug --sim_device cuda:2 --rl_device cuda:2 --num_envs 10 --seed 42 --max_iterations 100

nohup \
  python legged_gym/scripts/train.py \
  --task=g1_terrain\
  --resume \
   --experiment_name g1_terrain_exp \
   --run_name g1_terrain_7_15cm \
   --load_run Mar16_15-53-49_g1_terrain_0 \
   --checkpoint 8750 \
   --sim_device cuda:3 --rl_device cuda:3 \
   --num_envs 4096 --seed 42 --max_iterations 10000 \
   --headless \
   >g1_terrain_15.out 2>&1 &

