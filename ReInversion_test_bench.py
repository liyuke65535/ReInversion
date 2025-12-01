import os
import torch
import numpy as np
from PIL import Image
import time
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tools import seed_everything
from pipeline_ReInversion import ReInvFluxKontextPipeline


import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--npy_pth", type=str, default='test_bench/id_list.npy', help="Path to id list.")
    parser.add_argument("--gt_dir", type=str, default="test_bench/GT_3500", help="Path to source images.")
    parser.add_argument("--ref_dir", type=str, default="test_bench/Ref_3500", help="Path to reference images.")
    parser.add_argument("--mask_dir", type=str, default="test_bench/Mask_bbox_3500", help="Path to masks.")
    parser.add_argument("--res_dir", type=str, default='results/ReInversion', help="Path to result dir.")
    parser.add_argument("--model_name", type=str, default="black-forest-labs/FLUX.1-Kontext-dev", help="Model name.")
    parser.add_argument("--num_inference_steps", type=int, default=18, help="Inference steps.")
    parser.add_argument("--intermediate_step", type=int, default=4, help="Guided by source before it, guided by reference after it. (0 ~ num_inference_steps)")
    parser.add_argument("--eta", type=float, default=1.0, help="Weight for MSD. (0 ~ 1)")
    parser.add_argument("--height", type=int, default=512, help="Height of output.")
    parser.add_argument("--width", type=int, default=512, help="Width of output.")
    parser.add_argument("--determin_v", action="store_true", help="Use deterministic velocity for ReInversion.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    gt_dir = args.gt_dir
    ref_dir = args.ref_dir
    mask_dir = args.mask_dir

    pipe = ReInvFluxKontextPipeline.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")

    gt_files = {os.path.splitext(f)[0].replace('_GT', ''): f for f in os.listdir(gt_dir) if f.endswith('_GT.png')}
    ref_files = {os.path.splitext(f)[0].replace('_ref', ''): f for f in os.listdir(ref_dir) if f.endswith('_ref.png')}
    mask_files = {os.path.splitext(f)[0].replace('_mask', ''): f for f in os.listdir(mask_dir) if f.endswith('_mask.png')}

    common_keys = sorted(set(ref_files.keys()) & set(gt_files.keys()))
    print(f"Find {len(common_keys)} pairs.")

    pairs = []
    from tqdm import tqdm
    valid_pairs = np.load(args.npy_pth)
    for key in tqdm(valid_pairs):
        key = str(int(key)).zfill(12)
        gt_path = os.path.join(gt_dir, gt_files[key])
        ref_path = os.path.join(ref_dir, ref_files[key])
        mask_path = os.path.join(mask_dir, mask_files[key])
        
        pairs.append((key, gt_path, ref_path, mask_path))

    for k, gt, r, m in pairs[:3]:
        print(f"{k}: ref {r}, gt {gt}, mask {m}")

    print(f"Loaded {len(pairs)} pairs.")


    num_steps = args.num_inference_steps
    start = args.intermediate_step
    eta = args.eta
    h, w = args.height, args.width

    for i, item in enumerate(pairs):
        index = int(item[0])

        pth = f"{args.res_dir}/folder_{start}"
        if not os.path.exists(pth):
            os.makedirs(pth)
        save_pth = f"{pth}/output_{index}.png"

        input_img = Image.open(item[1]).resize([h, w])
        cond = Image.open(item[2]).resize([h, w])
        mask = Image.open(item[3]).resize([h, w])

        seed_everything(0)
        
        t0 = time.time()
        output_image = pipe(
            num_inference_steps=num_steps,
            image=input_img,
            image_2=cond,
            start_step=start,
            height=h,
            width=w,
            mask=mask,
            eta=eta,
            v_star=args.determin_v
        ).images[0]
        t1 = time.time()
        print(f"toal time: {(t1-t0):.3f}s")

        output_image.save(save_pth)
        print(f"save to {save_pth}.")