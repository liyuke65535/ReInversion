## Reversible Inversion for Training-Free Exemplar-guided Image Editing


To test on COCOEE from our model, you can use `ReInversion_test_bench.py`. But you should download the benchmark from "Paint-by-Example".

For example, 
```
CUDA_VISIBLE_DEVICES=1 \
python ReInversion_test_bench.py \
--npy_pth test_bench/id_list.npy \
--gt_dir test_bench/GT_3500 \
--ref_dir test_bench/Ref_3500 \
--mask_dir test_bench/Mask_bbox_3500 \
--res_dir results/ReInversion
```
or simply run:
```
sh run_test_bench.sh
```

Also, you can try examples in `demo.ipynb`. Or try `python ReInversion_demo.py`.

