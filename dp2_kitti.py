"""Depth Pro 2-L (Apple, research-only licence) depth maps for KITTI image_02 frames - reference B2, not the product.
The true focal length of image_02 (P_rect_02) is passed in, so DP2 does not have to guess it.
Run in ~/flatreal/.venv-dp2 from ~/metric3d.  usage: python dp2_kitti.py --drives 0020 [--out data/dp2]
Output: <out>/<drive>/<frame:010d>.npy (float16 metres, 1242x375) for run3d.py --depth-dir.
"""
import argparse, os, sys
from functools import partial
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from depth_pro_2 import DepthPro2, load_rgb
from depth_pro_2.model import to_normalized_tensor
from depth_pro_2.network.fov import FOVEstimator
import kitti

ap = argparse.ArgumentParser()
ap.add_argument("--drives", nargs="+", required=True)
ap.add_argument("--out", default="data/dp2")
a = ap.parse_args()
f_px = float(kitti.calib()["K"][0, 0])
model = DepthPro2.from_pretrained("apple/DepthPro2-L", precision=torch.bfloat16)
model.fov = FOVEstimator(torch.bfloat16)
model.fov.load_state_dict(load_file(hf_hub_download("apple/DepthPro2-focal", "model.safetensors")))
model = model.to("cuda").eval()
transform = partial(to_normalized_tensor, device=torch.device("cuda"))
for n in a.drives:
    src = os.path.join(kitti.drive_dir(n), "image_02", "data")
    dst = os.path.join(a.out, n)
    os.makedirs(dst, exist_ok=True)
    files = sorted(os.listdir(src))
    for k, fn in enumerate(files):
        p = os.path.join(dst, fn[:-4] + ".npy")
        if os.path.exists(p):
            continue
        image, _ = load_rgb(os.path.join(src, fn))
        with torch.no_grad():
            d = model.infer(transform(image), f_px=f_px)["depth"].float().cpu().numpy()
        d[~np.isfinite(d)] = 0  # sky / infinity -> invalid
        np.save(p, d.astype(np.float16))
    print(f"{n}: {len(files)} frames -> {dst}", flush=True)
print("done", a.out)
