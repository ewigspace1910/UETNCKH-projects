export LOCAL_PEFT_DIRECTORY=/home/ducanh/data/nimWS/loras
# mkdir $LOCAL_PEFT_DIRECTORY

# move a custom LoRA adapter to the local PEFT directory
cp -r /home/ducanh/data/nemoWS/results/models/ielts-writing-marking/llama38b-epoch=00-step=0999-val_loss=0.70-last $LOCAL_PEFT_DIRECTORY/llama3-lora-ielts-marking

chmod -R 777 $LOCAL_PEFT_DIRECTORY