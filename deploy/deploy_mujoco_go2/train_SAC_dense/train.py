import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from deploy.deploy_mujoco_go2.terrain_trainer import TerrainTrainer, TerrainGymEnv
from deploy.deploy_mujoco_go2.train_SAC_dense.dense_replay_buffer import FailureReplayBuffer
from deploy.deploy_mujoco_go2.train_SAC_dense.callbacks import DenseTrainingLogger


def train_sac_dense(
    go2_cfg,
    terrain_cfg,
    total_timesteps=50000,
    max_episode_steps=350,
    model_path="sac_dense_model.zip",
    log_dir="train_terrain_logs_dense",
    reward_threshold=8.0,
):
    os.makedirs(log_dir, exist_ok=True)

    def make_env():
        trainer = TerrainTrainer(go2_cfg, terrain_cfg)
        env = TerrainGymEnv(trainer, max_episode_steps=max_episode_steps)
        env = Monitor(env)
        return env

    vec_env = DummyVecEnv([make_env])

    model = SAC(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_starts=10000,
        replay_buffer_class=FailureReplayBuffer,
        replay_buffer_kwargs={"reward_threshold": reward_threshold},
    )

    callback = DenseTrainingLogger(out_dir=log_dir)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(model_path)
    return model_path


# TODO 修改model_path和log_dir
# TODO 训练时复制yaml文件
if __name__ == "__main__":
    go2_cfg = ["terrain", "go2.yaml"]
    terrain_cfg = "train_SAC_dense/terrain_config.yaml"
    train_sac_dense(
        go2_cfg,
        terrain_cfg,
        total_timesteps=100000,
        max_episode_steps=350,
        model_path="sac_dense_model.zip",
        log_dir="train_terrain_logs_dense",
        reward_threshold=8.0,
    )

