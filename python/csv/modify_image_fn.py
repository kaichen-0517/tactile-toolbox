import os
import pandas as pd

""" Modify the tactile_images column for the change from tg3 to tactile-bench.
    e.g. 0.png -> tactile_images/0.png
"""

def fix_targets_csv(root_dir="."):
    # for all targets.csv and targets.csv.bak
    targets = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename in ("targets.csv", "targets.csv.bak"):
                targets.append(os.path.join(dirpath, filename))

    for path in targets:
        df = pd.read_csv(path)
        if "tactile_images" not in df.columns:
            print(f"skip {path}")
            continue

        mask = ~df["tactile_images"].str.startswith("tactile_images/", na=True)
        df.loc[mask, "tactile_images"] = "tactile_images/" + df.loc[mask, "tactile_images"]

        df.to_csv(path, index=False)
        print(f"Modified: {path}")

fix_targets_csv(r"R:\optical_fibre_tg3\tactile_data_of\mg400")