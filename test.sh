#python legged_gym/scripts/play.py --task=go2 --experiment_name go2_exp --load_run Feb06_16-40-17_go2_exp_1 --num_envs 16 --seed 42 --max_iterations 10 --sim_device cuda:0 --rl_device cuda:1
#python legged_gym/scripts/play.py --task=g1 --experiment_name g1_exp --run_name g1_exp_2 --num_envs 16 --seed 42 --max_iterations 10 --sim_device cuda:0 --rl_device cuda:1
python train_terrain/test.py --task=go2_stair --experiment_name go2_stair_exp --num_envs 1 --seed 42 --max_iterations 10 --sim_device cuda:0 --rl_device cuda:1
