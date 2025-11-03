docker run  \
  --gpus all \
  --shm-size=2g \
  --net=host \
  --ulimit memlock=-1 \
  --rm -it \
  -v /home/ducanh/data/nemoWS:/workspace \
  -w /workspace \
  nvcr.io/nvidia/nemo:25.02 
