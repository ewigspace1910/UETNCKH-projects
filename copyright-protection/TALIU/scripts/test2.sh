# # CUDA_VISIBLE_DEVICES=0,1,2,3
# # CUDA_VISIBLE_DEVICES=0  \
# #  bash ./tools/dist_train.sh local_configs/metaseg/tiny/metaseg.tiny.512x512.ade.160k.py 1

# # python ./tools/train.py  local_configs/metaseg/tiny/metaseg.tiny.512x512.ade.160k.py \
# # python ./MetaSeg/xxx.py  "MetaSeg/local_configs/metaseg/tiny/metaseg.tiny.512x512.ade.160k.py" \
# #  --work-dir /home/k64t/xxxlogs/segment \
# #  --load-from "/home/k64t/Tampereddoc/zda_reimplementation/train/MetaSeg/pretrain/mscan_t.pth" \
# #  --gpu-id 0 --seed 123 \



for i in $(seq 160000 -4000 8000);
do
    echo "=======iter $i ======="
    python test.py  "/home/k64t/xxxlogs/segment/doctamper-2class-fullDA/metaseg.tiny.512x512.doctamper.py" \
    /home/k64t/xxxlogs/segment/doctamper-2class-fullDA/iter_$i.pth \
    --work-dir /home/k64t/xxxlogs/segment/doctamper-2class-fullDA \
    --gpu-id 0  --eval mIoU mFscore --sim-test --nm meta_$i \
    --show-dir "/home/k64t/Tampereddoc/zda_reimplementation/MetaSeg/logs/output"  >>  eval_log2.txt

    echo "=======iter $i - F ======="
    python test.py  "/home/k64t/xxxlogs/segment/doctamper-2class-fullDA/metaseg.tiny.512x512.doctamper-f.py" \
    /home/k64t/xxxlogs/segment/doctamper-2class-fullDA/iter_$i.pth \
    --work-dir /home/k64t/xxxlogs/segment/doctamper-2class-fullDA \
    --gpu-id 0  --eval mIoU mFscore --sim-test  --nm meta_$i \
    --show-dir "/home/k64t/Tampereddoc/zda_reimplementation/MetaSeg/logs/output"  >>  eval_log2.txt

    echo "=======iter $i - S ======="
    python test.py  "/home/k64t/xxxlogs/segment/doctamper-2class-fullDA/metaseg.tiny.512x512.doctamper-s.py" \
    /home/k64t/xxxlogs/segment/doctamper-2class-fullDA/iter_$i.pth \
    --work-dir /home/k64t/xxxlogs/segment/doctamper-2class-fullDA \
    --gpu-id 0  --eval mIoU mFscore --sim-test --nm meta_$i \
    --show-dir "/home/k64t/Tampereddoc/zda_reimplementation/MetaSeg/logs/output"  >>  eval_log2.txt

done



 





