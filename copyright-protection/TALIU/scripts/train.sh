# # CUDA_VISIBLE_DEVICES=0,1,2,3
# # CUDA_VISIBLE_DEVICES=0  \
# #  bash ./tools/dist_train.sh local_configs/metaseg/tiny/metaseg.tiny.512x512.ade.160k.py 1

# # python ./tools/train.py  local_configs/metaseg/tiny/metaseg.tiny.512x512.ade.160k.py \
# # python ./MetaSeg/xxx.py  "MetaSeg/local_configs/metaseg/tiny/metaseg.tiny.512x512.ade.160k.py" \
# #  --work-dir /home/k64t/xxxlogs/segment \
# #  --load-from "/home/k64t/Tampereddoc/zda_reimplementation/train/MetaSeg/pretrain/mscan_t.pth" \
# #  --gpu-id 0 --seed 123 \


# # python xxx.py  "local_configs/metaseg/base/metaseg.base.512x512.doctamper.py" \
# python xxx.py  "local_configs/metaseg/tiny/metaseg.tiny.512x512.doctamper.py" \
#  --work-dir /home/k64t/xxxlogs/segment/doctamper-2class-fullDA-coloradj \
#  --gpu-id 0 --seed 123 > logs/meta-coloradj.o

# python xxx.py  "local_configs/metaseg/tiny/metaseg.tiny.512x512.doctamper.py" \
#  --work-dir /home/k64t/xxxlogs/segment/doctamper-2class-fullDA-coloradj \
#  --gpu-id 0 --seed 123 --tamaug --tamaug-rate 0.05 > logs/meta-flipNcoloradj-aug.o


# python xxx.py  "local_configs/metaseg/tiny/metaseg.tiny.512x512.doctamper.py" \
#  --work-dir /home/k64t/xxxlogs/segment/doctamper-2class-fullDA-coloradj \
#  --gpu-id 0 --seed 123 --tamaug --tamaug-rate 0.1 > logs/meta-flipNcoloradj-aug.o


# python xxx.py  "local_configs/metaseg/tiny/metaseg.tiny.512x512.doctamper.py" \
#  --work-dir /home/k64t/xxxlogs/segment/doctamper-2class-fullDA-coloradj-agu015 \
#  --gpu-id 0 --seed 123 --tamaug --tamaug-rate 0.1 > logs/aug.log


python xxx.py  "local_configs/metaseg/tiny/metaseg.tiny.512x512.doctamper.py" \
 --work-dir /home/k64t/xxxlogs/segment/doctamper-2class-fullDA-coloradj-agu01 \
 --gpu-id 0 --seed 123 --tamaug --tamaug-rate 0.1 > logs/aug.log

# python xxx.py  "local_configs/metaseg/tiny/metaseg.tiny.512x512.doctamper.py" \
#  --work-dir /home/k64t/xxxlogs/segment/doctamper-2class-fullDA-coloradj-agu015 \
#  --gpu-id 0 --seed 123 --tamaug --tamaug-rate 0.2 > logs/aug.log







# python xxx.py  "local_configs/metaseg/tiny/DANK.tiny.512x512.doctamper.py" \
#  --work-dir /home/k64t/xxxlogs/segment/doctamper-DANKtest-coloradj \
#  --gpu-id 0 --seed 123 > logs/dank-coloradj.o


# python xxx.py  "local_configs/metaseg/tiny/DANK.tiny.512x512.doctamper.py" \
#  --work-dir /home/k64t/xxxlogs/segment/doctamper-DANK-aug_01 \
#  --gpu-id 0 --seed 123 --tamaug --tamaug-rate 0.1 > logs/dank_wtamaug_01.0


# python xxx.py  "local_configs/metaseg/tiny/DANK.tiny.512x512.doctamper.py" \
#  --work-dir /home/k64t/xxxlogs/segment/doctamper-DANK-aug_02 \
#  --gpu-id 0 --seed 123 --tamaug --tamaug-rate 0.2 > logs/dank_wtamaug_02.0


# python xxx.py  "local_configs/metaseg/tiny/DANK.tiny.512x512.doctamper.py" \
#  --work-dir /home/k64t/xxxlogs/segment/doctamper-DANK-aug_03 \
#  --gpu-id 0 --seed 123 --tamaug --tamaug-rate 0.3 > logs/dank_wtamaug_03.0 


# python xxx.py  "local_configs/metaseg/tiny/DANK.tiny.512x512.doctamper.py" \
#  --work-dir /home/k64t/xxxlogs/segment/doctamper-DANK-aug_005 \
#  --gpu-id 0 --seed 123 --tamaug --tamaug-rate 0.4 > logs/dank_wtamaug_005.0 



