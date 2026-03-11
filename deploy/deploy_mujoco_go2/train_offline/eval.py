import json
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import yaml

from deploy.deploy_mujoco_go2.train_offline.data_io import get_log_dirs, load_transitions_from_logs, stack_state_action
from deploy.deploy_mujoco_go2.train_offline.train_classifier import FailureClassifier, transition_label
from deploy.deploy_mujoco_go2.train_offline.train_ppo_offline import PPOActorCritic


def load_models(base_dir: str, log_name: str, device: torch.device):
    ppo_dir = os.path.join(base_dir, "train_logs", log_name, "ppo")
    cls_dir = os.path.join(base_dir, "train_logs", log_name, "classifier")

    ppo_ckpt_path = os.path.join(ppo_dir, "model_final.pt")
    cls_ckpt_path = os.path.join(cls_dir, "model_final.pt")

    if not os.path.exists(ppo_ckpt_path):
        raise FileNotFoundError(f"Missing PPO model: {ppo_ckpt_path}")
    if not os.path.exists(cls_ckpt_path):
        raise FileNotFoundError(f"Missing classifier model: {cls_ckpt_path}")

    ppo_ckpt = torch.load(ppo_ckpt_path, map_location=device)
    cls_ckpt = torch.load(cls_ckpt_path, map_location=device)

    return ppo_ckpt, cls_ckpt


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(base_dir, "train_config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    log_name = cfg.get("log_name", "offline_default")
    out_dir = os.path.join(base_dir, "train_logs", log_name, "eval")
    os.makedirs(out_dir, exist_ok=True)

    logs_cfg_path = os.path.join(base_dir, cfg.get("logs_config", "logs_config.yaml"))
    log_dirs, _ = get_log_dirs(logs_cfg_path)
    transitions = load_transitions_from_logs(log_dirs)

    states, actions = stack_state_action(transitions)
    if states.shape[0] == 0:
        raise RuntimeError("No transitions for offline eval.")

    obs_dim = int(states.shape[1])
    act_dim = int(actions.shape[1])

    device_cfg = cfg.get("device", "auto")
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)

    ppo_ckpt, cls_ckpt = load_models(base_dir, log_name, device)

    ppo = PPOActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden=256).to(device)
    ppo.load_state_dict(ppo_ckpt["model_state_dict"])
    ppo.eval()

    cls_input_dim = int(cls_ckpt["input_dim"])
    classifier = FailureClassifier(input_dim=cls_input_dim, hidden_dim=int(cfg["classifier"].get("hidden_dim", 256))).to(device)
    classifier.load_state_dict(cls_ckpt["model_state_dict"])
    classifier.eval()

    threshold = float(cfg["eval"].get("classifier_threshold", 0.55))
    use_action_in_obs = bool(cfg.get("classifier", {}).get("concat_action_to_obs", True))
    save_non_failure_trajectories = bool(cfg.get("eval", {}).get("save_non_failure_trajectories", False))
    rng = np.random.default_rng(int(cfg["eval"].get("random_seed", 42)))
    sample_limit = int(cfg["eval"].get("sample_limit", 0))

    n_total = states.shape[0] if sample_limit <= 0 else min(states.shape[0], sample_limit)

    used_ppo = 0
    used_random = 0
    p_fail_selected = []
    p_fail_ppo = []
    non_failure_samples = []

    with torch.no_grad():
        for i in range(n_total):
            s = states[i]
            s_t = torch.tensor(s[None, :], dtype=torch.float32, device=device)

            # Candidate PPO action
            dist = ppo.dist(s_t)
            a_ppo = torch.clamp(dist.mean, -1.0, 1.0).cpu().numpy().reshape(-1)

            if use_action_in_obs:
                feat_ppo = np.concatenate([s, a_ppo], axis=0).astype(np.float32)
            else:
                feat_ppo = s.astype(np.float32)
            feat_ppo_t = torch.tensor(feat_ppo[None, :], dtype=torch.float32, device=device)
            p_ppo = float(torch.sigmoid(classifier(feat_ppo_t)).item())
            p_fail_ppo.append(p_ppo)

            # Gate: if classifier says PPO is risky -> random
            if p_ppo > threshold:
                a_sel = rng.uniform(-1.0, 1.0, size=(act_dim,)).astype(np.float32)
                used_random += 1
            else:
                a_sel = a_ppo.astype(np.float32)
                used_ppo += 1

            if use_action_in_obs:
                feat_sel = np.concatenate([s, a_sel], axis=0).astype(np.float32)
            else:
                feat_sel = s.astype(np.float32)
            feat_sel_t = torch.tensor(feat_sel[None, :], dtype=torch.float32, device=device)
            p_sel = float(torch.sigmoid(classifier(feat_sel_t)).item())
            p_fail_selected.append(p_sel)

            if save_non_failure_trajectories and transition_label(transitions[i]) < 0.5:
                non_failure_samples.append(
                    {
                        "sample_idx": int(i),
                        "obs": np.asarray(s, dtype=np.float32).tolist(),
                        "selected_action": np.asarray(a_sel, dtype=np.float32).tolist(),
                        "pred_fail_selected": float(p_sel),
                    }
                )

    summary = {
        "num_samples": int(n_total),
        "classifier_threshold": threshold,
        "used_ppo": int(used_ppo),
        "used_random": int(used_random),
        "ratio_ppo": float(used_ppo / max(n_total, 1)),
        "ratio_random": float(used_random / max(n_total, 1)),
        "mean_pred_fail_selected": float(np.mean(p_fail_selected)) if len(p_fail_selected) > 0 else 0.0,
        "mean_pred_fail_ppo_only": float(np.mean(p_fail_ppo)) if len(p_fail_ppo) > 0 else 0.0,
        "concat_action_to_obs": bool(use_action_in_obs),
    }

    out_path = os.path.join(out_dir, "eval_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if save_non_failure_trajectories:
        pkl_path = os.path.join(out_dir, "non_failure_trajectories.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(non_failure_samples, f)
        print(f"Saved non-failure samples to: {pkl_path}")

    print(f"Saved offline eval summary to: {out_path}")
    print(summary)


if __name__ == "__main__":
    main()

