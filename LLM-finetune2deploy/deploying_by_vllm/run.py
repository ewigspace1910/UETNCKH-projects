from vllm import LLM
PATH="/home/ducanh/data/nimWS/loras/qwen/IWS-DSqwen25-7b-woRB-hf.bin"


llm = LLM(model=PATH, task="generate")  # Name or path of your model
llm.apply_model(lambda model: print(type(model)))