import json
import os

from deploy.deploy_mujoco_go2.train_offline.data_io import get_log_dirs, aggregate_failure_summary_csv


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    logs_cfg_path = os.path.join(base_dir, "logs_config.yaml")

    log_dirs, output_dir = get_log_dirs(logs_cfg_path)
    os.makedirs(output_dir, exist_ok=True)

    stats = aggregate_failure_summary_csv(log_dirs)

    json_path = os.path.join(output_dir, "aggregated_failure_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # One-row CSV output for quick spreadsheet import.
    csv_path = os.path.join(output_dir, "aggregated_failure_summary.csv")
    keys = list(stats.keys())
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        f.write(",".join(str(stats[k]) for k in keys) + "\n")

    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()

