from unsloth import FastLanguageModel
import torch
import pandas as pd
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
import json, re
fourbit_models = [
    "unsloth/Qwen3-1.7B-unsloth-bnb-4bit", # Qwen 14B 2x faster
    "unsloth/Qwen3-4B-unsloth-bnb-4bit",
    "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    "unsloth/Qwen3-14B-unsloth-bnb-4bit",
    "unsloth/Qwen3-32B-unsloth-bnb-4bit",

    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/Phi-4",
    "unsloth/Llama-3.1-8B",
    "unsloth/Llama-3.2-3B",
    "unsloth/orpheus-3b-0.1-ft-unsloth-bnb-4bit" # [NEW] We support TTS models!
] # More models at https://huggingface.co/unsloth
def extract_hash_answer(text:str):
    if "boxed{" in text:
        pattern = r"\\boxed{(.+)}"
        matches = re.findall(pattern, text)
        if len(matches) == 0: print(text)
        text = matches[0]
        return text
    newtext=""
    for i in text:
        if i.isdigit(): newtext += i
    if newtext.strip() == "":
        print("Error-->", text)
        return None
    return newtext.strip()
def generate_conversation(examples):
    problems  = [example["input"].strip() for example in examples]
    solutions = [example["output"].strip() for example in examples]
    points = [extract_hash_answer(example["output"].strip()) for example in examples]
    conversations = []
    for problem, solution, point in zip(problems, solutions, points):
        if point is None:
            print("\n--> SKIP output:", problem, solution)
            continue
        conversations.append([
            {"role" : "user",      "content" : problem},
            {"role" : "assistant", "content" : solution},
        ])
    return { "conversations": conversations, }
def read_jsonfile(file_path=""):
    all_samples = []
    # Open and read the file line by line
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)  # Parse the JSON object from each line
            # print(data)              # Do something with the JSON object
            all_samples += [data]
    return all_samples
def ft(model_type="unsloth/Qwen3-14B", model_name="lora_Qwen14b_simpleft", lora_targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",], 
       max_seq_length=2048, lora_rank=32, maxstep=600, number_epoch=None, 
       dpath="/home/ducanh/data/nemoWS/datasets/hey2/ielts-marking/raw_input4unsloth.json"):
    print("==============FT params===============")
    print(model_name, lora_targets, max_seq_length, lora_rank, maxstep, number_epoch, dpath)
    model, tokenizer = FastLanguageModel.from_pretrained(
        # model_name = "unsloth/Qwen3-14B",
        model_name = model_type,
        max_seq_length = max_seq_length,   # Context length - can be longer, but uses more memory
        load_in_4bit = True, # 4bit uses much less memory
        load_in_8bit = False,    # A bit more accurate, uses 2x memory
        full_finetuning = False, # We have full finetuning now!
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.85, # Reduce if out of memory
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
        target_modules = lora_targets, #could change?
        lora_alpha = lora_rank,  # Best to choose alpha = rank or rank*2
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 1234,
        use_rslora = False,   # We support rank stabilized LoRA
        loftq_config = None,  # And LoftQ
    )

    ielts_datsets = read_jsonfile(dpath)
    my_reasoning_conversations = tokenizer.apply_chat_template(
        generate_conversation(ielts_datsets)["conversations"],
        tokenize = False,
    )

    my_reasoning_set = pd.Series(my_reasoning_conversations)
    data = pd.concat([
        my_reasoning_set,
    ])
    data.name='text'

    combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
    combined_dataset = combined_dataset.shuffle(seed = 3407)
    print(combined_dataset[0].keys(), data )


#######################################
# Training

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = combined_dataset,
        eval_dataset = None, # Can set up evaluation!
        args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = 2, #shoudd be 2
            gradient_accumulation_steps = 4, # Use GA to mimic batch size!
            warmup_steps = 5,
            num_train_epochs = 2, # Set this for 1 full training run.
            max_steps = maxstep,
            learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            report_to = "none", # Use this for WandB etc
        ),
    )

    print("prepare done!!!!")
    import os
    os.environ['TORCHDYNAMO_VERBOSE']='1'
    trainer_stats = trainer.train()
    print("train ouput")
    print(trainer_stats)



    model.save_pretrained(f"/home/ducanh/nvidia-llm-pipeline/unslot/saves/{model_name}")  # Local saving
    tokenizer.save_pretrained(f"/home/ducanh/nvidia-llm-pipeline/unslot/saves/{model_name}")


if __name__== "__main__":
    dpaths = ["/home/ducanh/nvidia-llm-pipeline/unslot/ft_qwen/simple/data/train-new-task2-4qwen.json",
              "/home/ducanh/nvidia-llm-pipeline/unslot/ft_qwen/simple/data/train-new-task1-4qwen.json"
    ]
    for t in ['Qwen3-14B', 'Qwen3-8B']:
        for k in [1200, 2400, 3600]:
            for dpath, name in zip(dpaths, ['T1', 'T2']):
                ft(model_type=f"unsloth/{t}", model_name=f"lora_IELTS_writing{name}_s{k}_t{t}", lora_targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",], 
                    lora_rank=16, maxstep=k, dpath=dpath)