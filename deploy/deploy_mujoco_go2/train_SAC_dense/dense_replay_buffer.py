import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer


class FailureReplayBuffer(ReplayBuffer):
    """Replay buffer with failure-biased replay.

    Base behavior:
    - Always store incoming transitions (prevents empty buffer / sampling crash).

    Bias behavior:
    - Also store a second copy of full episode transitions if the episode is
      failure-relevant or cumulative reward is high.
    """

    def __init__(self, *args, reward_threshold=8.0, dense_sample_ratio=0.7, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_threshold = float(reward_threshold)
        self.dense_sample_ratio = float(np.clip(dense_sample_ratio, 0.0, 1.0))
        self._pending = [[] for _ in range(self.n_envs)]
        self._episode_reward = np.zeros((self.n_envs,), dtype=np.float32)
        # True means this row was inserted as dense-selected transition.
        self._dense_mask = np.zeros((self.buffer_size,), dtype=np.bool_)

    def _is_failure_info(self, info: dict) -> bool:
        if not isinstance(info, dict):
            return False
        return bool(
            info.get("fallen", False)
            # or info.get("collided", False)
            or info.get("base_collision", False)
            or info.get("thigh_collision", False)
            or info.get("stuck", False)
        )

    def _push_single_transition(self, tr):
        o, no, a, r, d, i = tr
        info = dict(i) if isinstance(i, dict) else {}
        info["dense_selected"] = True
        super().add(
            np.expand_dims(o, axis=0),
            np.expand_dims(no, axis=0),
            np.expand_dims(a, axis=0),
            np.array([r], dtype=np.float32),
            np.array([d], dtype=np.float32),
            [info],
        )
        inserted_idx = (self.pos - 1) % self.buffer_size
        self._dense_mask[inserted_idx] = True

    def add(self, obs, next_obs, action, reward, done, infos):
        # Always keep the base stream so SB3 can sample from step 1+.
        safe_infos = []
        if isinstance(infos, (list, tuple)):
            for info in infos:
                info_dict = dict(info) if isinstance(info, dict) else {}
                info_dict.setdefault("dense_selected", False)
                safe_infos.append(info_dict)
        else:
            safe_infos = [dict(dense_selected=False) for _ in range(self.n_envs)]

        super().add(obs, next_obs, action, reward, done, safe_infos)

        # Mark base stream as non-dense-selected.
        inserted_idx = (self.pos - 1) % self.buffer_size
        self._dense_mask[inserted_idx] = False

        # Keep episode-level cache and duplicate selected episodes for bias.
        obs = np.asarray(obs)
        next_obs = np.asarray(next_obs)
        action = np.asarray(action)
        reward = np.asarray(reward).reshape(-1)
        done = np.asarray(done).reshape(-1)

        for env_idx in range(self.n_envs):
            info = safe_infos[env_idx] if env_idx < len(safe_infos) else {}
            tr = (
                obs[env_idx].copy(),
                next_obs[env_idx].copy(),
                action[env_idx].copy(),
                float(reward[env_idx]),
                float(done[env_idx]),
                info,
            )
            self._pending[env_idx].append(tr)
            self._episode_reward[env_idx] += float(reward[env_idx])

            if bool(done[env_idx]):
                ep = self._pending[env_idx]
                has_failure = any(self._is_failure_info(t[5]) for t in ep)
                has_high_reward = self._episode_reward[env_idx] >= self.reward_threshold
                if has_failure or has_high_reward:
                    for t in ep:
                        self._push_single_transition(t)

                self._pending[env_idx] = []
                self._episode_reward[env_idx] = 0.0

    def sample(self, batch_size, env=None):
        """Prioritize dense-selected rows during sampling.

        - dense_sample_ratio controls fraction sampled from dense_selected=True rows.
        - fallback to uniform sampling if dense pool is empty.
        """
        upper_bound = self.buffer_size if self.full else self.pos
        if upper_bound <= 0:
            return super().sample(batch_size=batch_size, env=env)

        if self.dense_sample_ratio <= 0.0:
            return super().sample(batch_size=batch_size, env=env)

        dense_mask = self._dense_mask[:upper_bound]
        dense_pool = np.flatnonzero(dense_mask)
        if dense_pool.size == 0:
            return super().sample(batch_size=batch_size, env=env)

        n_dense = int(round(batch_size * self.dense_sample_ratio))
        n_dense = max(1, min(batch_size, n_dense))
        n_dense = min(n_dense, dense_pool.size)

        dense_inds = np.random.choice(dense_pool, size=n_dense, replace=False)

        n_other = batch_size - n_dense
        if n_other > 0:
            other_pool = np.flatnonzero(~dense_mask)
            if other_pool.size >= n_other:
                other_inds = np.random.choice(other_pool, size=n_other, replace=False)
            elif other_pool.size > 0:
                other_inds = np.random.choice(other_pool, size=n_other, replace=True)
            else:
                other_inds = np.random.randint(0, upper_bound, size=n_other)
            batch_inds = np.concatenate([dense_inds, other_inds])
        else:
            batch_inds = dense_inds

        np.random.shuffle(batch_inds)
        return self._get_samples(batch_inds, env=env)
