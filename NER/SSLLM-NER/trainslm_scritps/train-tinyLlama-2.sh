#SBATCH --nodes=1             # define node
#SBATCH --gpus-per-node=1     # define gpu limmit in 1 node
#SBATCH --ntasks=1            # define number tasks
#SBATCH --cpus-per-task=8    # There are 24 CPU cores
#SBATCH --time=7-23:00:00     # Max running time = 10 minutes
#SBATCH --output=output/train_bc2gm_jnlpba.o
#SBATCH --nodelist=node004
# Load module
# Some module avail:
source env/bin/activate


## pytorch-extra-py39-cuda11.2-gcc9
module load cuda11.2/toolkit/11.2.2
# module load pytorch-py39-cuda11.2-gcc9/1.9.1
# module load pytorch-extra-py39-cuda11.2-gcc9
module load opencv4-py39-cuda11.2-gcc9/4.5.4




litgpt finetune_lora checkpoints/TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --data JSON \
    --data.json_path ../datasets/slm4ner/BC2GM-train.json \
    --out_dir output/tinyllama/BC2GM \
    --data.val_split_fraction 0.2 \
    --data.prompt_style 'alpaca' \
    --train.epochs 8 \
    --logger_name csv \
    --precision "bf16-true" #32-true #If your GPU does not support bfloat16


litgpt finetune_lora checkpoints/TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --data JSON \
    --data.json_path ../datasets/slm4ner/JNLPBA-train.json \
    --out_dir output/tinyllama/JNLPBA \
    --data.val_split_fraction 0.2 \
    --data.prompt_style 'alpaca' \
    --train.epochs 8 \
    --logger_name csv \
    --precision "bf16-true" #32-true #If your GPU does not support bfloat16
