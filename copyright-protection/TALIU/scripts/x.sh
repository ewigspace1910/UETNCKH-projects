
# python train.py  "local_configs/metaseg/base/DANK.base.512x512.doctamper.py" \
#  --work-dir /home/k64t/xxxlogs/segment/doctamper-2class-DANK-B-new \
#  --gpu-id 0 --seed 123  >> logs/x.o 


# python train.py  "local_configs/metaseg/base/DANK2.base.512x512.doctamper.py" \
#  --work-dir /home/k64t/xxxlogs/segment/doctamper-2class-DANK2-B-new \
#  --gpu-id 0 --seed 123  >> logs/x.o 


# python train.py  "local_configs/metaseg/tiny/DANK.tiny.512x512.doctamper.py" \
# --work-dir /home/k64t/xxxlogs/segment/doctamper-2class-DANK-gray \
# --gpu-id 0 --seed 123 --tamaug --tamaug-rate 0.1 > logs/x.o 

python train.py  "local_configs/metaseg/tiny/DANK.tiny.512x512.doctamper.py" \
 --work-dir /home/k64t/xxxlogs/segment/doctamper-2class-DANK-gray \
 --gpu-id 0 --seed 123 --tamaug --tamaug-rate 0.1 > logs/x.o 


 
python xxx.py  "local_configs/metaseg/tiny/DANK2.tiny.512x512.doctamper.py" \
 --work-dir /home/k64t/xxxlogs/segment/doctamper-2class-newhead2W8 \
 --gpu-id 0 --seed 123 --tamaug --tamaug-rate 0.1 > logs/z.o #--tamaug --tamaug-rate 0.1