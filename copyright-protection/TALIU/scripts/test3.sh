# # CUDA_VISIBLE_DEVICES=0,1,2,3
# # CUDA_VISIBLE_DEVICES=0  \
# #  bash ./tools/dist_train.sh local_configs/metaseg/tiny/metaseg.tiny.512x512.ade.160k.py 1

# # python ./tools/train.py  local_configs/metaseg/tiny/metaseg.tiny.512x512.ade.160k.py \
# # python ./MetaSeg/xxx.py  "MetaSeg/local_configs/metaseg/tiny/metaseg.tiny.512x512.ade.160k.py" \
# #  --work-dir /home/k64t/xxxlogs/segment \
# #  --load-from "/home/k64t/Tampereddoc/zda_reimplementation/train/MetaSeg/pretrain/mscan_t.pth" \
# #  --gpu-id 0 --seed 123 \




# for i in $(seq 160000 -4000 80000);
# do
#     echo "=======iter $i ======="
#     python test.py  "/home/k64t/xxxlogs/segment/doctamper-2class-newhead2W8Wfuse/DANK2.tiny.512x512.doctamper.py" \
#     /home/k64t/xxxlogs/segment/doctamper-2class-newhead2W8Wfuse/iter_$i.pth \
#     --work-dir /home/k64t/xxxlogs/segment/doctamper-2class-newhead2W8Wfuse \
#     --gpu-id 0  --eval mIoU mFscore --sim-test --nm da$i \
#     --show-dir "/home/k64t/Tampereddoc/zda_reimplementation/MetaSeg/logs/output"  

#     echo "=======iter $i - F ======="
#     python test.py  "/home/k64t/xxxlogs/segment/doctamper-2class-newhead2W8Wfuse/DANK2.tiny.512x512.doctamper-f.py" \
#     /home/k64t/xxxlogs/segment/doctamper-2class-newhead2W8Wfuse/iter_$i.pth \
#     --work-dir /home/k64t/xxxlogs/segment/doctamper-2class-newhead2W8Wfuse \
#     --gpu-id 0  --eval mIoU mFscore --sim-test --nm da$i \
#     --show-dir "/home/k64t/Tampereddoc/zda_reimplementation/MetaSeg/logs/output"  

#     echo "=======iter $i - S ======="
#     python test.py  "/home/k64t/xxxlogs/segment/doctamper-2class-newhead2W8Wfuse/DANK2.tiny.512x512.doctamper-s.py" \
#     /home/k64t/xxxlogs/segment/doctamper-2class-newhead2W8Wfuse/iter_$i.pth \
#     --work-dir /home/k64t/xxxlogs/segment/doctamper-2class-newhead2W8Wfuse \
#     --gpu-id 0  --eval mIoU mFscore --sim-test --nm da$i \
#     --show-dir "/home/k64t/Tampereddoc/zda_reimplementation/MetaSeg/logs/output"  

# done

# for i in $(seq 160000 -4000 8000);
for i in 160000 148000 144000 152000 156000 140000 136000;
do
    echo "=======iter $i ======="
    python test.py  "/home/k64t/xxxlogs/segment/doctamper-2class-DANK-gray/DANK.tiny.512x512.doctamper.py" \
    /home/k64t/xxxlogs/segment/doctamper-2class-DANK-gray/iter_$i.pth \
    --work-dir /home/k64t/xxxlogs/segment/doctamper-2class-DANK-gray \
    --gpu-id 0  --eval mIoU mFscore --sim-test --nm new_da_$i \
    --show-dir "/home/k64t/Tampereddoc/zda_reimplementation/MetaSeg/logs/output"  

    # echo "=======iter $i - F ======="
    # python test.py  "/home/k64t/xxxlogs/segment/doctamper-2class-DANK-gray/DANK.tiny.512x512.doctamper-f.py" \
    # /home/k64t/xxxlogs/segment/doctamper-2class-DANK-gray/iter_$i.pth \
    # --work-dir /home/k64t/xxxlogs/segment/doctamper-2class-DANK-gray \
    # --gpu-id 0  --eval mIoU mFscore --sim-test  --nm new_da_$i \
    # --show-dir "/home/k64t/Tampereddoc/zda_reimplementation/MetaSeg/logs/output"  

    echo "=======iter $i - S ======="
    python test.py  "/home/k64t/xxxlogs/segment/doctamper-2class-DANK-gray/DANK.tiny.512x512.doctamper-s.py" \
    /home/k64t/xxxlogs/segment/doctamper-2class-DANK-gray/iter_$i.pth \
    --work-dir /home/k64t/xxxlogs/segment/doctamper-2class-DANK-gray \
    --gpu-id 0  --eval mIoU mFscore --sim-test --nm new_da_$i \
    --show-dir "/home/k64t/Tampereddoc/zda_reimplementation/MetaSeg/logs/output"  

done



 





