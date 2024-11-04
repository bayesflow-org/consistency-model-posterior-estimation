import os
import shutil


def to_id(method, architecture, num_train):
    return f"{method}-{architecture}-{num_train}"


out_dir = "figures_paper"
os.makedirs(out_dir, exist_ok=True)

filenames = ["main.pdf", "samples_main.pdf"]

for method in ["fmpe", "cmpe"]:
    for arch in ["naive", "unet"]:
        for num_train in [2000, 60000]:
            for fn in filenames:
                key = to_id(method, arch, num_train)
                shutil.copy(os.path.join("figures", key, fn), os.path.join(out_dir, f"{key}-{fn}"))
