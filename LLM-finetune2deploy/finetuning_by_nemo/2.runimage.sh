docker run -d \
  --gpus all \
  --shm-size=64g \
  --net=host \
  --ulimit memlock=-1 \
  --rm -it \
  -e XDG_CACHE_HOME=/workspace/.cache \
  -e HF_HOME=/workspace/.cache_hf \
  -e TRANSFORMERS_CACHE=/workspace/.cache_trans \
  -e NEMO_MODELS_CACHE=/workspace/.cache_nemo \
  -v /home/ducanh/data/nemoWS:/workspace \
  -w /workspace \
  nvcr.io/nvidia/nemo:25.04.nemotron-h
  # nvcr.io/nvidia/nemo:25.04.nemotron-h
  # nvcr.io/nvidia/nemo:25.04.rc2
  # nvcr.io/nvidia/nemo:25.02 

#docker ps --> find running image 
#to run jupyter notebook in docker -->  
#docker exec -it 4cbe1f80c836 huggingface-cli login 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
#docker exec -it 4cbe1f80c836 jupyter lab --ip 0.0.0.0 --port=8888 --allow-root
#docker exec -d 4cbe1f80c836 python ft-peft/terminal_script/finetuning/ft_wpert.py
#docker exec -it 4cbe1f80c836 python ft-peft/terminal_script/finetuning/ft_wpert_qwen.py
#docker exec -it 4cbe1f80c836 bash ft-peft/terminal_script/run_trt/run.sh
#docker exec -it 4cbe1f80c836 python  ft-peft/terminal_script/merging_lora2base.py
#docker exec -it 4cbe1f80c836 python  ft-peft/terminal_script/run_trt/for_trt/run_llm_deploy.py
#docker exec -it 4cbe1f80c836 python  ft-peft/terminal_script/run_trt/for_trt/run_fastapi.py
#docker exec -it 4cbe1f80c836 bash  ft-peft/terminal_script/run_trt/test_query.sh
#docker exec -it 4cbe1f80c836 python ft-peft/terminal_script/run_trt/query_pytrition.py
#docker exec -it 4cbe1f80c836 bash ft-peft/terminal_script/nemo2qnemo.sh 
#docker exec -it 4cbe1f80c836 bash ft-peft/terminal_script/datapreparation/get_r1_response.sh
#docker exec -it 4cbe1f80c836 nemo llm deploy nemo_checkpoint=/workspace/results/models/IWS-llama31/IWS-llama31-int4_awq triton_model_repository=/workspace/results/models/tmp_trt_llm dtype=int4_awq backend=trtllm max_output_len=400 max_input_len=4096 
#docker exec -it 4cbe1f80c836 /bin/bash
#export LD_LIBRARY_PATH=/usr/local/tensorrt/targets/x86_64-linux-gnu/lib:$LD_LIBRARY_PATH
#nvidia-ky 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

