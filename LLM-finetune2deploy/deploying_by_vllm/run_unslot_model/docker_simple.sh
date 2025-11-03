CACHE_HF=/home/ducanh/data/vllm/.cache_hf
HF_TOKEN=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
XDG_CACHE_HOME=/home/ducanh/data/vllm/.cache_xdg
LORAWEIHTS=/home/ducanh/nvidia-llm-pipeline/unslot/saves:/app/lora
WSPACE=/home/ducanh/data/vllm/.ws
docker run --rm -it --runtime nvidia --gpus all  --shm-size=16GB  \
    -v $CACHE_HF:/root/.cache/huggingface \
    -v $WSPACE:/vllm_workspace \
    -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
    -e XDG_CACHE_HOME=$XDG_CACHE_HOME \
    -w /vllm_workspace \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model meta-llama/Meta-Llama-3-8B --enable-lora \
    --tensor_parallel_size 1 \
    --served-model-name ducanh \
    --trust-remote-code 