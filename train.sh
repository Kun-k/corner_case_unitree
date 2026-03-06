#python legged_gym/scripts/train.py --task=go2_stair --experiment_name go2_terrain_0_target_exp --run_name go2_terrain_0_target_exp_0 --num_envs 4096 --seed 42 --max_iterations 10000 --sim_device cuda:2 --rl_device cuda:1 --headless

#nohup python legged_gym/scripts/train.py --task=go2_stair --experiment_name go2_terrain_0_exp --run_name go2_terrain_0_exp_0 --sim_device cuda:0 --rl_device cuda:0 --num_envs 4096 --seed 42 --max_iterations 10000 --headless >go2_terrain_0_exp_0.out 2>&1 &
#nohup python legged_gym/scripts/train.py --task=go2_stair --experiment_name go2_terrain_2_exp --run_name go2_terrain_2_exp_0 --sim_device cuda:1 --rl_device cuda:1 --num_envs 4096 --seed 42 --max_iterations 10000 --headless >go2_terrain_2_exp_0.out 2>&1 &
#nohup python legged_gym/scripts/train.py --task=go2_stair --experiment_name go2_terrain_4_exp --run_name go2_terrain_4_exp_0 --sim_device cuda:2 --rl_device cuda:2 --num_envs 4096 --seed 42 --max_iterations 10000 --headless >go2_terrain_4_exp_0.out 2>&1 &
#nohup python legged_gym/scripts/train.py --task=go2_stair --experiment_name go2_terrain_2_exp --run_name go2_terrain_2_exp_2 --sim_device cuda:0 --rl_device cuda:0 --num_envs 4096 --seed 42 --max_iterations 10000 --headless >go2_terrain_2_exp_2.out 2>&1 &
nohup python  legged_gym/scripts/train.py --task=go2_terrain_navigation --experiment_name go2_terrain_navigation_2_exp --run_name 0 --sim_device cuda:0 --rl_device cuda:0 --num_envs 4096 --seed 42 --max_iterations 10000 --headless >go2_terrain_2_exp_2.out 2>&1 &

#python legged_gym/scripts/train.py --task=go2_stair --experiment_name debug --run_name debug --sim_device cuda:0 --rl_device cuda:0 --num_envs 4096 --seed 42 --max_iterations 10000
#python legged_gym/scripts/train.py --task=go2_terrain_navigation --experiment_name debug --run_name debug --sim_device cuda:0 --rl_device cuda:0 --num_envs 4096 --seed 42 --max_iterations 10000
