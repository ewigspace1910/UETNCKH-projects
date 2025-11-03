CACHE=/home/ducanh/data/vllm/.cache
HF_TOKEN=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
# XDG_CACHE_HOME=/home/ducanh/data/vllm/.cache_xdg
LORAWEIHTS=/home/ducanh/nvidia-llm-pipeline/unslot/saves:/app/lora
WSPACE=/home/ducanh/data/vllm/.ws
docker run --rm -d --runtime nvidia --gpus all  --shm-size=16GB  \
    -v $CACHE:/root/.cache \
    -v $WSPACE:/vllm_workspace \
    -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
    -e VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 \
    -w /vllm_workspace \
    -p 8000\:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model unsloth/Qwen3-14B --enable-lora \
    --tensor_parallel_size 1 \
    --served-model-name ducanh-test \
    --trust-remote-code \
    --max_lora_rank 16 \
    --lora-modules ielts-speaking-gra=/vllm_workspace/loras/ielts-speaking-gra \
    --lora-modules ielts-speaking-lra=/vllm_workspace/loras/ielts-speaking-lra \

# 146.148.42.49:8000/docs
# docker exec -it 6ed430c96eee /bin/bash