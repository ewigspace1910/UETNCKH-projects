#!/bin/bash
rm /home/ubuntu/recsys/data/models/kt-model/*
cd /home/ubuntu/recsys/kt_model
source ~/miniconda3/etc/profile.d/conda.sh
conda activate monacobert
#!/bin/sh
current_date=$(date +%Y-%m-%d)
log_path="../data/logs/${current_date}-ktmodel.log"


# Run for training
echo "------pull raw data from db------" >> $log_path
python pull_raw_data.py --dbid 1 >> $log_path

echo "------preprocess raw data to dataset------" >> $log_path
python preprocess.py --dbid 1    >> $log_path

echo "------train kt-models------" >> $log_path
# python train_kt_model.py >> $log_path
python train_kt_model.py -M -s 1 -l 100 --dbid 1 >> $log_path
python train_kt_model.py -M -s 1 -l 180 --dbid 1 >> $log_path
python train_kt_model.py -M -s 2 -l 100 --dbid 1 >> $log_path
python train_kt_model.py -M -s 2 -l 180 --dbid 1 >> $log_path
python train_kt_model.py -M -s 3 -l 100 --dbid 1 >> $log_path
python train_kt_model.py -M -s 3 -l 180 --dbid 1 >> $log_path
python train_kt_model.py -M -s 5 -l 100 --dbid 1 >> $log_path
python train_kt_model.py -M -s 5 -l 180 --dbid 1 >> $log_path




echo "------update kt-model to s3 and db------" >> $log_path
python update_db.py    --dbid 1  >> $log_path