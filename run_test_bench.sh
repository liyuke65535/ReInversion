CUDA_VISIBLE_DEVICES=1 \
python ReInversion_test_bench.py \
--npy_pth test_bench/id_list.npy \
--gt_dir test_bench/GT_3500 \
--ref_dir test_bench/Ref_3500 \
--mask_dir test_bench/Mask_bbox_3500 \
--res_dir results/ReInversion