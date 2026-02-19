import yaml
import subprocess
import glob
import os
from datetime import datetime

MANIFEST_PATH = "rules/manifest.yaml"

def main():
    with open(MANIFEST_PATH, "r") as f:
        manifest = yaml.safe_load(f)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")

    for niche_name, niche_data in manifest["niches"].items():
        input_pattern = niche_data["input_glob"]
        config_path = niche_data["config"]
        output_dir = os.path.join(
            manifest["defaults"]["outputs_root"],
            niche_data["output_subdir"]
        )

        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f"{timestamp}.csv")

        input_files = glob.glob(input_pattern, recursive=True)

        if not input_files:
            print(f"[SKIP] No input files found for {niche_name}")
            continue

        print(f"[RUNNING] {niche_name}")
        print(f"Found {len(input_files)} files")

        cmd = [
            "python",
            "kcp_pipeline.py",
            "--in",
            input_pattern,
            "--out",
            output_file,
            "--config",
            config_path,
        ]

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"[ERROR] Failed for {niche_name}")
        else:
            print(f"[DONE] Output saved to {output_file}")

if __name__ == "__main__":
    main()
