
python test.py  "configs/swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py" \
 /home/k64t/xxxlogs/segment/doctamper-2class-fullDA/iter_8000.pth \
 --work-dir /home/k64t/xxxlogs/segment/doctamper-swin \
 --gpu-id 0 \
 --eval mIoU mFscore --sim-test --nm swin \
    --show-dir "/home/k64t/Tampereddoc/zda_reimplementation/MetaSeg/logs/output"


# python xxx.py  "local_configs/metaseg/tiny/DANK2.tiny.512x512.doctamper.py" \
#  --work-dir /home/k64t/xxxlogs/segment/doctamper-2class-newhead2W8 \
#  --gpu-id 0 --seed 123 #--tamaug --tamaug-rate 0.1 > logs/z.o #--tamaug --tamaug-rate 0.1

# python xxx.py  "local_configs/metaseg/tiny/DANK2.tiny.512x512.doctamper.py" \
#  --work-dir /home/k64t/xxxlogs/segment/doctamper-2class-newhead2W8Wfuse \
#  --gpu-id 0 --seed 123 --tamaug --tamaug-rate 0.1 > logs/z.o