from unsloth import FastLanguageModel
import torch
import pandas as pd
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer, SFTTrainer, SFTConfig
import json, re
import re
reasoning_start = "<think>"
reasoning_end   = "</think>"
solution_start = "<CONCLUSION>"
solution_end = "</CONCLUSION>"

system_prompt = \
f"""You are a IELTS Speaking Examiner.
Analysis question and corresponding student response then provide your comment.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide the score from 0-9 between {solution_start}{solution_end}"""


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
    
    if "</think>" in text: text=text.split("</think>")[-1]
    newtext=""
    for i in text:
        if i.isdigit(): newtext += i
    if newtext.strip() == "":
        print("Error-->", text)
        return None
    return newtext.strip()

def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Match if format is seen exactly!
        if match_format.search(response) is not None: score += 3.0
        scores.append(score)
    return scores
match_format = re.compile(
    rf"^[\s]{{0,}}"\
    rf"{reasoning_start}.+?{reasoning_end}.*?"\
    rf"{solution_start}(.+?){solution_end}"\
    rf"[\s]{{0,}}$",
    flags = re.MULTILINE | re.DOTALL
)
def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Count how many keywords are seen - we penalize if too many!
        # If we see 1, then plus some points!
        score += 0.5 if response.count(reasoning_start) == 1 else -1.0
        score += 0.5 if response.count(reasoning_end)   == 1 else -1.0
        score += 0.5 if response.count(solution_start)  == 1 else -1.0
        score += 0.5 if response.count(solution_end)    == 1 else -1.0
        scores.append(score)
    return scores

def check_answer(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1)
        if (guess := match_format.search(r)) is not None else None \
        for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(0)
            continue
        # Correct answer gets 3 points!
        if guess == true_answer:
            score += 3.0
        # Match if spaces are seen, but less reward
        elif guess.strip() == true_answer.strip():
            score += 1.5
        else:
            # We also reward it if the answer is close via ratios!
            # Ie if the answer is within some range, reward it!
            try:
                ratio = float(guess) / float(true_answer)
                if   ratio >= 0.9 and ratio <= 1.1: score += 1.0
                elif ratio >= 0.8 and ratio <= 1.2: score += 0.5
                else: score -= 1.5 # Penalize wrong answers
            except:
                score -= 1.5 # Penalize
        scores.append(score)
    return scores

def check_length(prompts, completions, answer, **kwargs):
    max_len=800
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = []
    for r in responses:
        try:
            extracted_responses.append(len(r.split(" ")))
        except: extracted_responses.append(None)

    scores = []
    for guess in zip(extracted_responses):
        score = 0
        if guess is None:
            scores.append(0)
            continue
        # Correct answer gets 3 points!
        if guess == max_len:
            score += 2.0
        # Match if spaces are seen, but less reward
        else:
            # We also reward it if the answer is close via ratios!
            # Ie if the answer is within some range, reward it!
            try:
                ratio = float(guess) / float(max_len)
                if   ratio >= 0.8 and ratio <= 1.2: score += 1.0
                elif ratio >= 0.5 and ratio <= 1.5: score += 0.5
                else: score -= 1.5 # Penalize wrong answers
            except:
                score -= 1.5 # Penalize
        scores.append(score)
    return scores

def generate_conversation(examples):
    problems  = [example["input"].strip() for example in examples]
    solutions = [example["output"].strip() for example in examples]
    points = [extract_hash_answer(example["output"].strip()) for example in examples]
    conversations = []
    for problem, solution, point in zip(problems, solutions, points):
        if point is None:
            # print("\n--> SKIP output:", problem, solution)
            continue
        x = {
            "prompt" : [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": problem},
            ],
            "answer": point, 'question': problem
        }
        conversations.append(x)
    return conversations

def read_jsonfile(file_path="/home/ducanh/data/nemoWS/datasets/hey2/ielts-marking/raw_input4unsloth4qrpo.json"):
    all_samples = []
    # Open and read the file line by line
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)  # Parse the JSON object from each line
            # print(data)              # Do something with the JSON object
            all_samples += [data]
    return all_samples


import re
global PRINTED_TIMES
PRINTED_TIMES = 0
global PRINT_EVERY_STEPS
PRINT_EVERY_STEPS = 50
match_numbers = re.compile(
    solution_start + r".*?([\d\.\,]{1,})",
    flags = re.MULTILINE | re.DOTALL
)
def check_numbers(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1)
        if (guess := match_numbers.search(r)) is not None else None \
        for r in responses
    ]

    scores = []
    # Print only every few steps
    global PRINTED_TIMES
    global PRINT_EVERY_STEPS
    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        print("\n\n",'*'*20, f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    PRINTED_TIMES += 1

    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(0)
            continue
        # Convert to numbers
        try:
            true_answer = float(true_answer.strip())
            # Remove commas like in 123,456
            guess       = float(guess.strip().replace(",", ""))
            scores.append(1.5 if guess == true_answer else -0.5)
        except:
            scores.append(0)
            continue
    return scores



def ft(model_name="lora_Qwen14b_qrpo", lora_targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",], 
       max_seq_length=2048, lora_rank=32, maxstep=600, max_prompt_length=400, number_epoch=None, dpath=""):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen3-14B",
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
    my_reasoning_conversations = generate_conversation(ielts_datsets)    


    training_args = GRPOConfig(
        learning_rate = 5e-5,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "adamw_torch_fused",
        logging_steps = 1,
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4, # Increase to 4 for smoother training
        num_generations = 4, # Decrease if out of memory
        max_prompt_length = max_prompt_length,
        max_completion_length = max_seq_length - max_prompt_length,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps = maxstep,
        save_steps = 50,
        max_grad_norm = 0.1,
        report_to = "none", # Can use Weights & Biases
        output_dir = f"/home/ducanh/nvidia-llm-pipeline/unslot/ft_qwen/qrpo/outputs/{model_name}",
    )
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            match_format_exactly,
            match_format_approximately,
            check_answer,
            check_numbers,
            check_length
        ],
        args = training_args,
        train_dataset = my_reasoning_conversations,
    )
    trainer.train()
    model.save_pretrained(f"/home/ducanh/nvidia-llm-pipeline/unslot/saves/{model_name}")  # Local saving
    tokenizer.save_pretrained(f"/home/ducanh/nvidia-llm-pipeline/unslot/saves/{model_name}")


if __name__== "__main__":
    dpath=""
    dpath = "/home/ducanh/data/nemoWS/datasets/hey2/ielts-marking/new_raw_input4unsloth4qrpo_gra.json"
    ft(model_name="lora_Qwen14b_qrpo_full16_ogra", lora_targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",], 
       lora_rank=16, maxstep=800, dpath=dpath)
    ft(model_name="lora_Qwen14b_qrpo_qkvo16_ogra", lora_targets=["q_proj", "k_proj", "v_proj", "o_proj",],
        lora_rank=16, maxstep=800, dpath=dpath)