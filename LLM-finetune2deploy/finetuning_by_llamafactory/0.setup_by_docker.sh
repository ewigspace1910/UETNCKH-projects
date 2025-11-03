#docker build -t llamafactory -f LLaMA-Factory/docker/docker-cuda/Dockerfile .
#run the docker
# Run the Docker container
docker run -d \
  --gpus all \
  --shm-size=64g \
  --net=host \
  --ulimit memlock=-1 \
  --rm -it \
  -v /home/ducanh/data/llfactoryWS/.cache:/root/.cache/\
  -v /home/ducanh/data/llfactoryWS/data:/app/data \
  -v /home/ducanh/data/llfactoryWS/output:/app/output \
  -v /home/ducanh/data/llfactoryWS/code:/app/code \
  -w /app \
  --name llamafactory-container \
  llamafactory bash


#docker exec -it 223bcf4adc98 /bin/bash
#docker exec -d 223bcf4adc98 huggingface-cli login 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
#docker excel -d 223bcf4adc98 llamafactory-cli train output/train_llama4.json
#docker excel -d 223bcf4adc98 bash code/ft_ll4.bash
#docker exec -it 223bcf4adc98 llamafactory-cli webui
#docker exec -d 223bcf4adc98 bash code/ft_qwen3_32b.sh
#docker exec -d 223bcf4adc98 bash code/ft_qwq_32b.sh
#docker exec -d 223bcf4adc98 bash code/ft_phi4_14b.sh
