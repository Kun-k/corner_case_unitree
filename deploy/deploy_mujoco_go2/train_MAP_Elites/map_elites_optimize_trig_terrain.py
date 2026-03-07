import argparse
import json
import os
from dataclasses import dataclass

import numpy as np

from deploy.deploy_mujoco_go2.terrain_trainer import TerrainTrainer


@dataclass
class MAPElitesConfig:
    dim: int
    bins_x: int
    bins_y: int
    desc_min_x: float
    desc_max_x: float
    desc_min_y: float
    desc_max_y: float
    seed: int = 0


class MAPElitesArchive:
    """2D MAP-Elites archive with maximization objective."""

    def __init__(self, cfg: MAPElitesConfig):
        self.cfg = cfg
        self.rng = np.random.RandomState(cfg.seed)

        self.fitness = np.full((cfg.bins_x, cfg.bins_y), -np.inf, dtype=np.float64)
        self.vectors = np.full((cfg.bins_x, cfg.bins_y, cfg.dim), np.nan, dtype=np.float64)
        self.desc = np.full((cfg.bins_x, cfg.bins_y, 2), np.nan, dtype=np.float64)
        self.done = np.zeros((cfg.bins_x, cfg.bins_y), dtype=np.bool_)
        self.count = 0

    def _bin_index(self, dx, dy):
        x = (dx - self.cfg.desc_min_x) / (self.cfg.desc_max_x - self.cfg.desc_min_x + 1e-12)
        y = (dy - self.cfg.desc_min_y) / (self.cfg.desc_max_y - self.cfg.desc_min_y + 1e-12)
        ix = int(np.clip(np.floor(x * self.cfg.bins_x), 0, self.cfg.bins_x - 1))
        iy = int(np.clip(np.floor(y * self.cfg.bins_y), 0, self.cfg.bins_y - 1))
        return ix, iy

    def add(self, vector, reward, descriptor, done):
        dx, dy = float(descriptor[0]), float(descriptor[1])
        ix, iy = self._bin_index(dx, dy)
        if reward > self.fitness[ix, iy]:
            if not np.isfinite(self.fitness[ix, iy]):
                self.count += 1
            self.fitness[ix, iy] = reward
            self.vectors[ix, iy] = vector
            self.desc[ix, iy] = np.array([dx, dy], dtype=np.float64)
            self.done[ix, iy] = bool(done)
            return True, (ix, iy)
        return False, (ix, iy)

    def random_elite(self):
        mask = np.isfinite(self.fitness)
        ids = np.argwhere(mask)
        if len(ids) == 0:
            return None
        k = self.rng.randint(0, len(ids))
        ix, iy = ids[k]
        return self.vectors[ix, iy].copy(), (int(ix), int(iy))

    def best(self):
        if self.count == 0:
            return None
        idx = np.unravel_index(np.argmax(self.fitness), self.fitness.shape)
        ix, iy = int(idx[0]), int(idx[1])
        return {
            "idx": [ix, iy],
            "reward": float(self.fitness[ix, iy]),
            "vector": self.vectors[ix, iy].copy(),
            "descriptor": self.desc[ix, iy].copy(),
            "done": bool(self.done[ix, iy]),
        }

    def coverage(self):
        return float(self.count) / float(self.cfg.bins_x * self.cfg.bins_y)

    def qd_score(self):
        mask = np.isfinite(self.fitness)
        if not np.any(mask):
            return 0.0
        return float(np.sum(self.fitness[mask]))


def decode_params_to_angle_array(x, mode_y, mode_x):
    """Map vector -> angle_array[my,mx,2] = [theta, amplitude_weight]."""
    n = mode_y * mode_x
    theta = np.reshape(x[:n], (mode_y, mode_x))
    alt_raw = np.reshape(x[n : 2 * n], (mode_y, mode_x))

    theta = np.mod(theta, 2.0 * np.pi)
    alt = np.tanh(alt_raw)
    angle_array = np.stack([theta, alt], axis=-1)
    return angle_array.astype(np.float32)


def sample_random_vector(rng, mode_y, mode_x):
    n = mode_y * mode_x
    theta = rng.uniform(0.0, 2.0 * np.pi, size=(n,))
    alt_raw = rng.normal(0.0, 1.0, size=(n,))
    return np.concatenate([theta, alt_raw], axis=0)


def mutate_vector(rng, parent, mode_y, mode_x, sigma_theta, sigma_alt):
    n = mode_y * mode_x
    child = parent.copy()
    child[:n] += rng.normal(0.0, sigma_theta, size=(n,))
    child[n:] += rng.normal(0.0, sigma_alt, size=(n,))
    child[:n] = np.mod(child[:n], 2.0 * np.pi)
    return child


def compute_descriptors(terrain_h):
    """Behavior descriptors from generated terrain (2D).

    d0: height std
    d1: mean gradient magnitude
    """
    h = np.asarray(terrain_h, dtype=np.float32)
    gx = np.gradient(h, axis=1)
    gy = np.gradient(h, axis=0)
    grad_mag = np.sqrt(gx * gx + gy * gy)

    d0 = float(np.std(h))
    d1 = float(np.mean(grad_mag))
    return np.array([d0, d1], dtype=np.float32)


def evaluate_candidate(trainer, angle_array, max_robot_steps, safe_radius_m, blend_radius_m):
    trainer.reset()

    trainer.terrain_changer.generate_trig_terrain(angle_array)
    trainer.terrain_changer.enforce_safe_spawn_area(
        center_world=(0.0, 0.0),
        safe_radius_m=safe_radius_m,
        blend_radius_m=blend_radius_m,
        target_height=0.0,
    )

    terrain_snapshot = np.array(trainer.terrain_changer.hfield, dtype=np.float32)
    descriptor = compute_descriptors(terrain_snapshot)

    if trainer.render:
        trainer.viewer.update_hfield(trainer.terrain_changer.hfield_id)
        trainer.viewer.sync()

    total_reward = 0.0
    done = False

    for _ in range(max_robot_steps):
        _, _, reward, done, _ = trainer.step_only_robot()
        total_reward += float(reward)
        if done:
            break

    return total_reward, descriptor, done


def export_archive(archive, mode_y, mode_x, out_dir):
    np.save(os.path.join(out_dir, "archive_fitness.npy"), archive.fitness)
    np.save(os.path.join(out_dir, "archive_vectors.npy"), archive.vectors)
    np.save(os.path.join(out_dir, "archive_descriptors.npy"), archive.desc)
    np.save(os.path.join(out_dir, "archive_done.npy"), archive.done)

    best = archive.best()
    if best is not None:
        best_angle_array = decode_params_to_angle_array(best["vector"], mode_y, mode_x)
        np.save(os.path.join(out_dir, "best_angle_array.npy"), best_angle_array)
        np.save(os.path.join(out_dir, "best_vector.npy"), best["vector"])


def main():
    parser = argparse.ArgumentParser(description="MAP-Elites search for trig terrain parameters.")
    parser.add_argument("--go2-task", type=str, default="terrain", help="Go2 controller task folder")
    parser.add_argument("--go2-config", type=str, default="go2.yaml", help="Go2 config file name")
    parser.add_argument("--terrain-config", type=str, default="terrain_config.yaml", help="Terrain trainer config file")

    parser.add_argument("--mode-y", type=int, default=10)
    parser.add_argument("--mode-x", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--max-robot-steps", type=int, default=350)
    parser.add_argument("--safe-radius-m", type=float, default=1.0)
    parser.add_argument("--blend-radius-m", type=float, default=2.0)

    parser.add_argument("--bins-x", type=int, default=20)
    parser.add_argument("--bins-y", type=int, default=20)
    parser.add_argument("--desc-min-x", type=float, default=0.0)
    parser.add_argument("--desc-max-x", type=float, default=0.2)
    parser.add_argument("--desc-min-y", type=float, default=0.0)
    parser.add_argument("--desc-max-y", type=float, default=0.3)

    parser.add_argument("--init-random-ratio", type=float, default=0.35)
    parser.add_argument("--sigma-theta", type=float, default=0.35)
    parser.add_argument("--sigma-alt", type=float, default=0.45)

    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--out-dir", type=str, default="deploy/deploy_mujoco_go2/train_MAP_Elites/logs/map_elites_logs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    trainer = TerrainTrainer([args.go2_task, args.go2_config], f"train_MAP_Elites/{args.terrain_config}")
    trainer.init_skip_time = 0
    trainer.init_skip_frame = 10

    dim = 2 * args.mode_y * args.mode_x
    archive = MAPElitesArchive(
        MAPElitesConfig(
            dim=dim,
            bins_x=args.bins_x,
            bins_y=args.bins_y,
            desc_min_x=args.desc_min_x,
            desc_max_x=args.desc_max_x,
            desc_min_y=args.desc_min_y,
            desc_max_y=args.desc_max_y,
            seed=args.seed,
        )
    )

    history = []

    try:
        for it in range(args.iterations):
            elite = archive.random_elite()
            use_random = (elite is None) or (rng.rand() < args.init_random_ratio)

            if use_random:
                x = sample_random_vector(rng, args.mode_y, args.mode_x)
                parent_bin = None
            else:
                parent, parent_bin = elite
                x = mutate_vector(rng, parent, args.mode_y, args.mode_x, args.sigma_theta, args.sigma_alt)

            angle_array = decode_params_to_angle_array(x, args.mode_y, args.mode_x)
            reward, descriptor, done = evaluate_candidate(
                trainer,
                angle_array,
                max_robot_steps=args.max_robot_steps,
                safe_radius_m=args.safe_radius_m,
                blend_radius_m=args.blend_radius_m,
            )

            inserted, cell = archive.add(x, reward, descriptor, done)

            best = archive.best()
            row = {
                "iteration": it,
                "reward": float(reward),
                "descriptor": [float(descriptor[0]), float(descriptor[1])],
                "inserted": bool(inserted),
                "cell": [int(cell[0]), int(cell[1])],
                "parent_bin": parent_bin,
                "coverage": archive.coverage(),
                "qd_score": archive.qd_score(),
                "best_reward": None if best is None else float(best["reward"]),
            }
            history.append(row)

            print(
                f"[it {it:04d}] reward={reward:.3f} desc=({descriptor[0]:.4f},{descriptor[1]:.4f}) "
                f"inserted={inserted} coverage={archive.coverage():.3f} qd={archive.qd_score():.3f}"
            )

            if (it + 1) % args.save_every == 0:
                export_archive(archive, args.mode_y, args.mode_x, args.out_dir)
                with open(os.path.join(args.out_dir, "history.json"), "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2)

        export_archive(archive, args.mode_y, args.mode_x, args.out_dir)

        summary = {
            "iterations": args.iterations,
            "coverage": archive.coverage(),
            "qd_score": archive.qd_score(),
            "mode_y": args.mode_y,
            "mode_x": args.mode_x,
            "bins_x": args.bins_x,
            "bins_y": args.bins_y,
            # "best": archive.best(),
        }

        with open(os.path.join(args.out_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Done. coverage={archive.coverage():.3f}, qd_score={archive.qd_score():.3f}, logs={args.out_dir}")

    finally:
        trainer.close_viewer()


if __name__ == "__main__":
    main()


"""
nohup python deploy/deploy_mujoco_go2/train_MAP_Elites/map_elites_optimize_trig_terrain.py \
  --go2-task terrain \
  --go2-config go2.yaml \
  --terrain-config terrain_config.yaml \
  --iterations 1000 \
  --mode-y 10 \
  --mode-x 10   >emaes.out 2>&1 &
"""
