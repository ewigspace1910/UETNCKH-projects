

# Linear Regression
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd
import json, re
from unsloth import FastLanguageModel
import torch
reasoning_start = "<think>"
reasoning_end   = "</think>"
solution_start = "<CONCLUSION>"
solution_end = "</CONCLUSION>"
system_prompt = \
f"""You are a IELTS Speaking Examiner.
Analysis question and corresponding student response then provide your comment.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide the score from 0-9 between {solution_start}{solution_end}"""

def read_jsonfile(file_path="/home/ducanh/data/nemoWS/datasets/hey2/ielts-marking/raw_input4unsloth.json"):
    all_samples = []
    # Open and read the file line by line
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)  # Parse the JSON object from each line
            # print(data)              # Do something with the JSON object
            all_samples += [data]
    return all_samples
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

import re

def extract_score(sentence):
    match = re.search(r'(\d+(\.\d+)?)', sentence)
    return float(match.group(0)) if match else None


def get_score(text:str):
    try:
        text = text.replace("\n", "").replace("\\n", "").replace("</CONCLUSION>", "").replace("*", "").strip()[-100:]
        print("---",text, "----", end="\t")
        print(extract_score(text))
        return extract_score(text)
    except:
        print("Error om convert:", text)
        return 0

def infer(model, tokenizer, messages, mlo=500):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True, # Must add for generation
        enable_thinking = False, # Disable thinking
    )
    
    output_tokens = model.generate(
        **tokenizer(text, return_tensors = "pt").to("cuda"),
        max_new_tokens = mlo, # Increase for longer outputs!
        temperature = 1.0, top_p = 0.95, top_k = 20, # For thinking
        # streamer = TextStreamer(tokenizer, skip_prompt = True),
    )
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    get_score(output_text)
    return output_text  # Return the generated text instead of token IDs

def get_linest(y_values_poly, x_values_poly):
    # Convert to NumPy arrays and reshape for sklearn
    X_poly = np.array(x_values_poly).reshape(-1, 1)
    Y_poly = np.array(y_values_poly)

    # Define the polynomial degree (degree 2 for quadratic fit)
    degree = 1

    # Transform the data to include polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_poly_transformed = poly.fit_transform(X_poly)

    # Fit the polynomial regression model
    model_poly = LinearRegression()
    model_poly.fit(X_poly_transformed, Y_poly)

    return model_poly.coef_, model_poly.intercept_

def get_mapping(coefs, intercept, x_values):
    return [max(1,min(9,round(intercept+sum([coef*x_val**degree for degree, coef in enumerate(coefs)])))) for x_val in x_values]

def test(model_path="/home/ducanh/nvidia-llm-pipeline/unslot/saves/lora_Qwen14b_qrpo", mlo=1000, lora=16):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path ,
        max_seq_length = 2048,   # Context length - can be longer, but uses more memory
        load_in_4bit = True, # 4bit uses much less memory
        load_in_8bit = False,    # A bit more accurate, uses 2x memory
        full_finetuning = False, # We have full finetuning now!
        max_lora_rank = lora,
        gpu_memory_utilization = 0.85, # Reduce if out of memory
        # token = "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",      # use one if using gated models
    )

    gtests = read_jsonfile("/home/ducanh/nvidia-llm-pipeline/unslot/ft_qwen/test_gra.json")
    predictions = []
    for i in gtests:
        messages = [{"role": "system", "content": system_prompt}, 
                    {"role" : "user", "content" : i['input']}]
        output = i['output']
        predict = infer(model, tokenizer, messages,mlo=mlo).split("</think>")[-1].strip()
        predictions.append({"label":output, "pred":predict})
        print({"label":output, "pred":predict})   
    with open(f"{model_path}/gap.json", "w", encoding="utf-8") as test_f:
        for sample in predictions:
            json.dump(sample, test_f)
            test_f.write("\n")

    # ltests = read_jsonfile("/home/ducanh/data/nemoWS/datasets/hey2/ielts-marking/speaking/ielts_speech_synthetic_data_cttr_080525_test_lr.jsonl")
    # predictions = []
    # for i in ltests:
    #     messages = [{"role" : "user", "content" : i['input']}]
    #     output = i['output']
    #     predict = infer(model, tokenizer,  messages, mlo=mlo).split("</think>")[-1].strip()
    #     predictions.append({"label":output, "pred":predict})
    #     print({"label":output, "pred":predict})   
        
    # with open(f"{model_path}/lrp.json", "w", encoding="utf-8") as test_f:
    #     for sample in predictions:
    #         json.dump(sample, test_f)
    #         test_f.write("\n")

    try:
        # Assuming the JSONL file has 'input', 'label', and 'prediction' fields
        gra_data = read_jsonfile(f"{model_path}/gap.json")
        # gra_input = [gra_item['label'] for gra_item in gra_data]
        gra_target = [float(gra_item['label'].split(" ")[-1]) if "label" in gra_item else float(gra_item['input'].split(" ")[-1]) for gra_item in gra_data]
        gra_preds = [get_score(gra_item['pred']) for gra_item in gra_data]
        gra_linest = get_linest(gra_target, gra_preds)

        # lr_data = read_jsonfile(f"{model_path}/lrp.json")
        # # lr_input = [lr_item['input'] for lr_item in lr_data]
        # lr_target = [float(lr_item['label'].split(" ")[-1]) for lr_item in lr_data]
        # lr_preds = [get_score(lr_item['pred']) for lr_item in lr_data]
        # # Prediction score extraction end
        # lr_linest = get_linest(lr_target, lr_preds)

        gra_df = pd.DataFrame({
            # "input": gra_input,
            "target": gra_target,
            "pred": gra_preds,
            "acc":[item_target-1<=item<=item_target+1 for item_target,item in zip(gra_target,gra_preds)],
            "pred_mapped": get_mapping(gra_linest[0], gra_linest[1], gra_preds),
            "acc_mapped": [item_target-1<=item<=item_target+1 for item_target,item in zip(gra_target,get_mapping(gra_linest[0], gra_linest[1], gra_preds))],
        })

        # lr_df = pd.DataFrame({
        #     # "input": lr_input,
        #     "target": lr_target,
        #     "pred": lr_preds,
        #     "acc":[item_target-1<=item<=item_target+1 for item_target,item in zip(lr_target,lr_preds)],
        #     "pred_mapped": get_mapping(lr_linest[0], lr_linest[1], lr_preds),
        #     "acc_mapped": [item_target-1<=item<=item_target+1 for item_target,item in zip(lr_target,get_mapping(lr_linest[0], lr_linest[1], lr_preds))],
        # })
        
        print(f"Grammatical Range and Accuracy mapping coef and intercept:\nCoefficients: {gra_linest[0]}\nIntercept: {gra_linest[1]}")
        print(f"Grammatical Range and Accuracy rubric accuracy:\nUnmapped: {round(gra_df['acc'].sum()/gra_df['acc'].count()*100,2)}%\nMapped: {round(gra_df['acc_mapped'].sum()/gra_df['acc_mapped'].count()*100,2)}%")
        print()
        # print()
        # print(f"Lexical Resource mapping coef and intercept:\nCoefficients: {lr_linest[0]}\nIntercept: {lr_linest[1]}")
        # print(f"Lexical Resource rubric accuracy:\nUnmapped: {round(lr_df['acc'].sum()/lr_df['acc'].count()*100,2)}%\nMapped: {round(lr_df['acc_mapped'].sum()/lr_df['acc_mapped'].count()*100,2)}%")

        with open(f"{model_path}/log.txt", "w") as file:
            file.write(f"Grammatical Range and Accuracy mapping coef and intercept:\nCoefficients: {gra_linest[0]}\nIntercept: {gra_linest[1]}\n\n")
            file.write(f"Grammatical Range and Accuracy rubric accuracy:\nUnmapped: {round(gra_df['acc'].sum()/gra_df['acc'].count()*100,2)}%\nMapped: {round(gra_df['acc_mapped'].sum()/gra_df['acc_mapped'].count()*100,2)}%\n\n")
            
            # file.write(f"Lexical Resource mapping coef and intercept:\nCoefficients: {lr_linest[0]}\nIntercept: {lr_linest[1]}\n\n")
            # file.write(f"Lexical Resource rubric accuracy:\nUnmapped: {round(lr_df['acc'].sum()/lr_df['acc'].count()*100,2)}%\nMapped: {round(lr_df['acc_mapped'].sum()/lr_df['acc_mapped'].count()*100,2)}%\n")
    except Exception as e: 
        print(e)     
        print("\n\n\t\t=================") 
        
if __name__ == "__main__":
    import time
    paths = [
        "/home/ducanh/nvidia-llm-pipeline/unslot/ft_qwen/qrpo/outputs/lora_Qwen14b_qrpo_full16_ogra/checkpoint-350",
        # "/home/ducanh/nvidia-llm-pipeline/unslot/ft_qwen/qrpo/outputs/lora_Qwen14b_qrpo_full16_ogra/checkpoint-300",
        # "/home/ducanh/nvidia-llm-pipeline/unslot/ft_qwen/qrpo/outputs/lora_Qwen14b_qrpo_full16_ogra/checkpoint-250",
        # "/home/ducanh/nvidia-llm-pipeline/unslot/ft_qwen/qrpo/outputs/lora_Qwen14b_qrpo_qkvo16/checkpoint-600",
        # "/home/ducanh/nvidia-llm-pipeline/unslot/ft_qwen/qrpo/outputs/lora_Qwen14b_qrpo_qkvo16/checkpoint-550",
        # "/home/ducanh/nvidia-llm-pipeline/unslot/ft_qwen/qrpo/outputs/lora_Qwen14b_qrpo_full16_ogra/checkpoint-150",
    ]
    for path in paths:
        print("\n\n")
        print("\t\t===="*2)
        print("load model from ", path)
        s = time.time()
        test(path, mlo=2000)
        print("---Total time is ", round(time.time()-s, 2), "----")