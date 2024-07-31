#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate monacobert
current_date=$(date +%Y-%m-%d)
log_path="../data/logs/test/${current_date}.test"
cd /home/ubuntu/recsys/kt_model >> $log_path
pwd >> $log_path
pip list >> $log_path
echo "---------------${current_date}---------------------" >> $log_path



