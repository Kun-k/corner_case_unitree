#python legged_gym/scripts/train.py --task=go2 --experiment_name go2_exp --run_name go2_exp_2 --num_envs 4096 --seed 42 --max_iterations 10000 --sim_device cuda:2 --rl_device cuda:1 --headless
#python legged_gym/scripts/train.py --task=g1 --experiment_name g1_exp --run_name g1_exp_2 --num_envs 4096 --seed 42 --max_iterations 10000 --sim_device cuda:2 --rl_device cuda:1 --headless
python legged_gym/scripts/train.py --task=go2_stair --experiment_name go2_stair_exp --run_name go2_stair_exp_2 --num_envs 4096 --seed 42 --max_iterations 10000 --sim_device cuda:2 --rl_device cuda:1 --headless
