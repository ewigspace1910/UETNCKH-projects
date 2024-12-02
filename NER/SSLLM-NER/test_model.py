
import argparse
import transformers as ts
from transformers import DataCollatorForTokenClassification

from datasets import Dataset
from datasets import load_metric
import time
import sys
import json
import argparse
import os
import numpy as np
import pandas as pd
import csv
from tqdm import tqdm
from omegaconf import OmegaConf
import transformers
from transformers import AutoModelForCausalLM
from litgpt import LLM
from utils import calculate_precision, calculate_recall, calculate_f1_score

def load_and_preprocess_dataset(args, tokenizer):
    def load_ner_dataset(folder):
        allLabels = set(pd.read_csv(os.path.join(folder, "train.tsv"), sep="\t",
                                    header=None, quoting=csv.QUOTE_NONE, encoding='utf-8')[1])

        label_to_index = {label: index for index,
                          label in enumerate(allLabels)}
        index_to_label = {index: label for index,
                          label in enumerate(allLabels)}

        def load_subset(subset):
            lines = []

            with open(os.path.join(folder,subset), mode="r") as f:
                lines = f.readlines()

            sentences = []
            labels = []

            currentSampleTokens = []
            currentSampleLabels = []

            for line in lines:
                if line.strip() == "":
                    sentences.append(currentSampleTokens)
                    labels.append(currentSampleLabels)
                    currentSampleTokens = []
                    currentSampleLabels = []
                else:
                    cleanedLine = line.replace("\n", "")
                    token, label = cleanedLine.split(
                        "\t")[0].strip(), cleanedLine.split("\t")[1].strip()
                    currentSampleTokens.append(token)
                    currentSampleLabels.append(label_to_index[label])

            dataDict = {
                "tokens": sentences,
                "ner_tags": labels,
            }

            return Dataset.from_dict(dataDict)

        trainingDataset = load_subset("train.tsv")
        validationDataset = Dataset.from_dict(
            load_subset("train_dev.tsv")[len(trainingDataset):])
        testDataset = load_subset("test.tsv")

        return {
            "train": trainingDataset,
            "validation": validationDataset,
            "test": testDataset,
            "all_ner_tags": list(allLabels),
        }

    dataset = load_ner_dataset(args.datasetPath)
    label_names = dataset["all_ner_tags"]

    # Get the values for input_ids, token_type_ids, attention_mask
    def tokenize_adjust_labels(all_samples_per_split):
        tokenized_samples = tokenizer.batch_encode_plus(
            all_samples_per_split["tokens"], is_split_into_words=True, max_length=512)
        total_adjusted_labels = []

        for k in range(0, len(tokenized_samples["input_ids"])):
            prev_wid = -1
            word_ids_list = tokenized_samples.word_ids(batch_index=k)
            existing_label_ids = all_samples_per_split["ner_tags"][k]
            i = -1
            adjusted_label_ids = []

            for wid in word_ids_list:
                if(wid is None):
                    adjusted_label_ids.append(-100)
                elif(wid != prev_wid):
                    i = i + 1
                    adjusted_label_ids.append(existing_label_ids[i])
                    prev_wid = wid
                else:
                    adjusted_label_ids.append(existing_label_ids[i])

            total_adjusted_labels.append(adjusted_label_ids)

        tokenized_samples["labels"] = total_adjusted_labels

        return tokenized_samples

    tokenizedTrainDataset = dataset["train"].map(
        tokenize_adjust_labels, batched=True)
    tokenizedValDataset = dataset["validation"].map(
        tokenize_adjust_labels, batched=True)
    tokenizedTestDataset = dataset["test"].map(
        tokenize_adjust_labels, batched=True)

    return tokenizedTrainDataset, tokenizedValDataset, tokenizedTestDataset, label_names




def convert(args, data):
    sentence = " ".join(data['tokens'])
    tag_sent = " ".join([args.label_names[n] for n in data['ner_tags'] ])
    entities = []
    tag_pair = []
    tmp_word = ""
    for (t, n) in zip(data['tokens'], data['ner_tags']):
        n= args.label_names[n] # O I B
        tag_pair += [(t, n)]
        if n == "O":
            if tmp_word != "": 
                entities += [tmp_word.strip()]
                tmp_word  = ""
        else:
            tmp_word += f"{t} "
        
    transformed_data = []
    # Typ1 : Convert to simple form
    answer= "; ".join([f"[{e}] " for idx, e in enumerate(entities)])  
    answer+= f"\n There are {len(entities)} entities in the sentence" if len(entities) > 1 else f"\n There is 1 entity in the sentence"
    transformed_data.append({
            "instruction": f"You are a help full asistant to perform the following task. The task is to identify the name of entities belong to {args.type} from given sentences.",
            "input": f"{sentence}",
            "output": answer
        })
    # transformed_data.append({
    #         "instruction": f"Help me identify and classifying named entities belong to {args.type} from given sentences and tag each word in sentence with one of labels as [I-B-O]. B (Beginning): Indicates that the token is at the beginning of a named entity. I (Inside): Indicates that the token is inside a named entity but not at the beginning. O (Outside): Indicates that the token is outside of any named entity.",
    #         "input": f"For each word in the sentence:  ```{sentence}```, please tag them to one of labels as (I-B-O)",
    #         "output": f"Labels for words in ```{sentence}``` are following : \n " + " ".join([label for word, label in tag_pair])
    #     })

    return transformed_data, tag_pair, entities, tag_sent


def _generate_binary_vector_(s1_words, s2_words):
    """
Problem: Given 2 sentences like this:
sentence_1 : [SOS] clustering of missense mutations in the ataxia - telangiectasia gene in a sporadic t - cell leukaemia . [EOS]
sentence_2 : [SOS] clustering of missense mutations in the @ gene in a @ . [EOS]

Output:   a Binary marking vector to indicate words in sentence_1 which are hidden by '@' in sentence 2. 
For example in the case above, sentence_2 have 2 '@'. the first used to hide the words `ataxia - telangiectasia` while the second represents 'sporadic t - cell leukaemia'
The output vector should be : [0 0 0 0 0 0 0 1 1 1 0 0 0 0 1 1 1 1 1 0 0]
    """
    markers = [0]*len(s1_words) # Initialize a marker vector with 0s
    i = 0
    j = 0
    
    while i < len(s1_words) and j < len(s2_words):
        if s2_words[j] == "@": # We've found a placeholder
            # Advance in s1 until the next normal word in s2 is found
            while i < len(s1_words) and s1_words[i] != s2_words[j+1]: 
                # Mark these words as hidden
                markers[i] = 1
                # print("===>marking:", i, s1_words[i], j, s2_words[j+1] )
                i+=1
            j+=1
        else: 
            i+=1
            j+=1
            
    return markers

def encode_output2IOB(input, output):
    pad_input  = ["[SOS]"] + input.lower().split() + ["[EOS]"]

    pad_output = ["[SOS]"] + output.lower().split()+ ["[EOS]"]
    pad_output = [a  for i, a in enumerate(pad_output) if not (a == '@' and pad_output[i-1] == '@' and i > 0) ] 
    
    # print(">",pad_input)
    # print("=", pad_output)
    IOB = _generate_binary_vector_(pad_input, pad_output)

    return IOB[1:-1]
 



def main(args):
    tokenizer = ts.AutoTokenizer.from_pretrained(args.tokenizerPath)
    tokenizedTrainDataset, tokenizedValDataset, tokenizedTestDataset,  label_names = load_and_preprocess_dataset(args, tokenizer)
    args.label_names = label_names
    train_data = [data for data in tokenizedTrainDataset] + [data for data in tokenizedValDataset]
    train_json_data, test_json_data = [], []
    llm = LLM.load(args.model)
    predictions, labels = [], []
    for idx, data in enumerate(tokenizedTestDataset): 
        transformed_data, tag_pair, entities, tag_sent = convert(args, data)
        question =  transformed_data[0]['input']    
        text_out = llm.generate(question, top_k=1, max_new_tokens=100)
        text_out = text_out.split("\n")[0]
       # for k, v in transformed_data[0].items():
       #     print("\t--->", k, ": ", v)
        IOB_prediction = encode_output2IOB(input=question, output=text_out)
        biLabel = [0 if i == 'O' else 1 for i in tag_sent.split()]
        if len(IOB_prediction) < len(biLabel): IOB_prediction += [0]* (len(biLabel)-len(IOB_prediction))
        elif len(IOB_prediction) > len(biLabel): IOB_prediction = IOB_prediction[:len(biLabel)+1]
        predictions += IOB_prediction
        labels     += biLabel
        #print("\n-", question, "\n+", text_out)
        #print(IOB_prediction)
        #print(biLabel)
        #print("============================\n\n")
        #if idx > 10: break

    #calculate presion ,recall, F1
    precision = calculate_precision(prediction=predictions, label=labels) 
    recall    = calculate_recall(prediction=predictions, label=labels)
    f1_score  = calculate_f1_score(prediction=predictions, label=labels)

    print("\n\n==============================================================")
    print("\t Dataset: ", args.dataset, "\t| model: ", args.model)
    print("\t-Precision: ",precision)
    print("\t-Recall   : ",recall)
    print("\t-F1score  : ",f1_score)
    print("\n\n==============================================================\n\n")




        

        
    # Alternative, stream the response one token at a time will be cooler :>:
    # result = llm.generate("hi", stream=True)
    # for e in result:
    #     print(e, end="", flush=True)

__available_model__ = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "microsoft/phi-2",
    "codellama/CodeLlama-7b-hf", "stabilityai/stablecode-completion-alpha-3b"
    "stabilityai/stablelm-base-alpha-3b",
    "openlm-research/open_llama_13b", "openlm-research/open_llama_3b" 
    "EleutherAI/pythia-31m", "keeeeenw/MicroLlama"
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DANK1910!LLM playground")
    # data
    import os
    parser.add_argument("--model", "-m", type=str, default="name model or path to your local model", required=True)
    parser.add_argument("-i", "--init", type=str)   
    parser.add_argument("-d", "--dataset", type=str, choices=["NCBI-disease", "BC5CDR-disease", "BC2GM", "JNLPBA", "BC4CHEMD", "BC5CDR-chem", "s800", "linnaeus"], required=True)   
    parser.add_argument("-t", "--type", type=str, choices=["Disease", "Gene/protein", "Drug/chem", "Species"], required=True)   
    parser.add_argument("-p", "--datafolderPath", type=str,required=True)   
    args = parser.parse_args()
    initial_time = time.time()
    args.tokenizerPath = "nlpie/distil-biobert"
    datasetName = args.dataset
    args.datasetPath = os.path.join(args.datafolderPath, datasetName)
    main(args)
    print("\n\n\t\t\t =====================The running took ", (time.time() -initial_time)/3600, " hours==============================")


    # insent = "with sporadic T - cell prolymphocytic leukaemia ( T - PLL ) , a rare clonal malignancy with similarities to a mature T - cell leukaemia seen in A - T , we demonstrate a high frequency of ATM mutations in T - PLL"
    # outsent= "with @ @ @ @ @ ( @ @ @ ) , a rare @ @ with similarities to a mature @ @ @ seen in @ @ @ ."
    # encode_output2IOB(insent, outsent)
