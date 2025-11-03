#following this instruction: https://docs.nvidia.com/nim/large-language-models/latest/peft.html
export LOCAL_PEFT_DIRECTORY=/home/ducanh/data/nimWS/loras/llama
export NIM_PEFT_SOURCE=/home/nvs/loras
export NIM_PEFT_REFRESH_INTERVAL=3600   # will check NIM_PEFT_SOURCE for newly added models every hour
export CONTAINER_NAME='llama'
export NIM_CACHE_PATH='/home/ducanh/data/nimWS/.cache'
export NGC_API_KEY='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'
# mkdir -p "$NIM_CACHE_PATH"
# chmod -R 777 $NIM_CACHE_PATH
#docker login nvcr.io #Username: $oauthtoken
docker run -it --rm --name=$CONTAINER_NAME \
    --runtime=nvidia \
    --gpus all \
    --shm-size=16GB \
    -e NCCL_DEBUG=INFO \
    -e NGC_API_KEY=$NGC_API_KEY \
    -e NIM_PEFT_SOURCE \
    -e NIM_PEFT_REFRESH_INTERVAL \
    -v $NIM_CACHE_PATH:/opt/nim/.cache \
    -v $LOCAL_PEFT_DIRECTORY:$NIM_PEFT_SOURCE \
    -p 8000:8000 \
    nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8.3



# docker login nvcr.io 
# 