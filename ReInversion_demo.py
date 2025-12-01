import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import torch
from PIL import Image

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tools import seed_everything
from pipeline_ReInversion import ReInvFluxKontextPipeline




if __name__ == "__main__":


    pipe = ReInvFluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16
    )
    pipe = pipe.to("cuda")



    res = 512

    src_img = Image.open("assets/labubu_1.png").convert("RGB").resize([res, res])
    ref_img = Image.open("assets/labubu_3.png").convert("RGB").resize([res, res])
    mask = Image.open("assets/labubu_1_mask.png").convert("L").resize([res, res])

    seed_everything(0)
    output_image = pipe(
        num_inference_steps=28,
        image=src_img,
        image_2=ref_img,
        start_step=3,
        height=res,
        width=res,
        mask=mask,
    ).images[0]
    dir = "ReInv_results"
    if not os.path.exists(dir):
        os.makedirs(dir)
    save_path = os.path.join(dir, f"output.png")
    output_image.save(save_path)
    print(f"save to {save_path}")