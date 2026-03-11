import csv
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import yaml


def _resolve_path(base_dir: str, p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(base_dir, p))


def load_logs_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("logs_config must be a dict yaml")
    return cfg


def get_log_dirs(logs_config_path: str) -> Tuple[List[str], str]:
    cfg = load_logs_config(logs_config_path)
    base_dir = os.path.dirname(os.path.abspath(logs_config_path))
    log_dirs_cfg = cfg.get("log_dirs", [])
    if not isinstance(log_dirs_cfg, list):
        raise ValueError("log_dirs must be a list")
    log_dirs = [_resolve_path(base_dir, p) for p in log_dirs_cfg]
    output_dir = _resolve_path(base_dir, cfg.get("output_dir", "./train_logs/offline_stats"))
    return log_dirs, output_dir


def _load_pickle_file(pkl_path: str):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _extract_chains_from_obj(obj) -> List[List[Dict]]:
    chains: List[List[Dict]] = []

    if isinstance(obj, dict):
        if "chain" in obj and isinstance(obj["chain"], list):
            chains.append(obj["chain"])
        elif "chains" in obj and isinstance(obj["chains"], list):
            for c in obj["chains"]:
                if isinstance(c, list):
                    chains.append(c)

    elif isinstance(obj, list):
        if len(obj) == 0:
            return chains
        # list of episodes dict
        if isinstance(obj[0], dict) and "chain" in obj[0]:
            for ep in obj:
                c = ep.get("chain", [])
                if isinstance(c, list):
                    chains.append(c)
        # list of transition dict
        elif isinstance(obj[0], dict) and "obs" in obj[0] and "action" in obj[0]:
            chains.append(obj)

    return chains


def load_transitions_from_logs(log_dirs: List[str]) -> List[Dict]:
    transitions: List[Dict] = []
    for d in log_dirs:
        pkl_path = os.path.join(d, "collision_failures.pkl")
        if not os.path.exists(pkl_path):
            continue
        try:
            obj = _load_pickle_file(pkl_path)
            chains = _extract_chains_from_obj(obj)
            for chain in chains:
                for tr in chain:
                    if isinstance(tr, dict) and "obs" in tr and "action" in tr:
                        transitions.append(tr)
        except Exception:
            continue
    return transitions


def _read_last_csv_row(csv_path: str) -> Dict:
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) == 0:
        return {}
    return rows[-1]


def aggregate_failure_summary_csv(log_dirs: List[str]) -> Dict[str, float]:
    keys = [
        "episodes_evaluated",
        "total_failures",
        "collision_failures",
        "fall_failures",
        "base_collision_failures",
        "thigh_collision_failures",
        "stuck_failures",
    ]
    agg = {k: 0.0 for k in keys}

    for d in log_dirs:
        csv_path = os.path.join(d, "failure_summary.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            row = _read_last_csv_row(csv_path)
        except Exception:
            continue
        for k in keys:
            if k in row and row[k] != "":
                try:
                    agg[k] += float(row[k])
                except Exception:
                    pass

    total = max(agg["episodes_evaluated"], 1.0)
    for k in keys:
        if k == "episodes_evaluated":
            continue
        agg[f"prob_{k}"] = float(agg[k] / total)

    return agg


def stack_state_action(transitions: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    states = []
    actions = []
    for tr in transitions:
        s = np.asarray(tr.get("obs", []), dtype=np.float32)
        a = np.asarray(tr.get("action", []), dtype=np.float32)
        if s.size == 0 or a.size == 0:
            continue
        states.append(s)
        actions.append(a)
    if len(states) == 0:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)
    return np.stack(states, axis=0), np.stack(actions, axis=0)


def build_transition_arrays(transitions: List[Dict]):
    states = []
    actions = []
    dones = []
    infos = []
    for tr in transitions:
        s = np.asarray(tr.get("obs", []), dtype=np.float32)
        a = np.asarray(tr.get("action", []), dtype=np.float32)
        if s.size == 0 or a.size == 0:
            continue
        states.append(s)
        actions.append(a)
        dones.append(bool(tr.get("done", False)))
        infos.append(tr.get("info", {}))

    if len(states) == 0:
        return {
            "states": np.zeros((0, 0), dtype=np.float32),
            "actions": np.zeros((0, 0), dtype=np.float32),
            "dones": np.zeros((0,), dtype=np.bool_),
            "infos": [],
        }

    return {
        "states": np.stack(states, axis=0),
        "actions": np.stack(actions, axis=0),
        "dones": np.asarray(dones, dtype=np.bool_),
        "infos": infos,
    }

