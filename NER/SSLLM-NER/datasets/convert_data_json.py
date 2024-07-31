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





def load_and_preprocess_dataset(args, tokenizer):
    def load_ner_dataset(folder):
        allLabels = set(pd.read_csv(folder + "train.tsv", sep="\t",
                                    header=None, quoting=csv.QUOTE_NONE, encoding='utf-8')[1])

        label_to_index = {label: index for index,
                          label in enumerate(allLabels)}
        index_to_label = {index: label for index,
                          label in enumerate(allLabels)}

        def load_subset(subset):
            lines = []

            with open(folder + subset, mode="r") as f:
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
    entities = []
    tag_pair = []
    tmp_word = ""
    n_ent    = 0
    for (t, n) in zip(data['tokens'], data['ner_tags']):
        n= args.label_names[n] # O I B
        tag_pair += [(t, n)]
        if n == "O":
            # if tmp_word != "": 
            #     entities += [tmp_word.strip()]
            #     tmp_word  = ""
            tmp_word += f"{t} "
        else:
            tmp_word += f"@ "
            if n == 'B': n_ent += 1
        
    transformed_data = []
    # Typ1 : Convert to simple form
    # answer= "; ".join([f"[{e}] " for idx, e in enumerate(entities)])  
    answer = tmp_word
    answer+= f"\n There are {n_ent} entities in the sentence" if n_ent > 1 else f"\n There is {n_ent} entity in the sentence"
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

    # Typ2 : extra questions : MLM/..."the i-th word is??? "
    # [
    #     {
    #         "instruction": "Arrange the given numbers in ascending order.",
    #         "input": "what is ???", #Extract word sth ...
    #         "output": "0, 2, 3, 4, 8"
    #     },
    #     ...
    # ]

    # Typ3 : Word finding....
    # [
    #     {
    #         "instruction": "Arrange the given numbers in ascending order.",
    #         "input": "xxx is the name or belong the name of diease?",
    #         "output": "0, 2, 3, 4, 8"
    #     },
    #     ...
    # ]
    return transformed_data


def main(args):
    tokenizer = ts.AutoTokenizer.from_pretrained(args.tokenizerPath)
    tokenizedTrainDataset, tokenizedValDataset, tokenizedTestDataset,  label_names = load_and_preprocess_dataset(args, tokenizer)
    args.label_names = label_names
    train_data = [data for data in tokenizedTrainDataset] + [data for data in tokenizedValDataset]
    train_json_data, test_json_data = [], []
    for data in tqdm(train_data): 
        transformed_data = convert(args, data)
        train_json_data += transformed_data

  
    for data in tqdm(tokenizedTestDataset): 
        transformed_data = convert(args, data)
        test_json_data += transformed_data



    #save
    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)
    with open(f'{args.savepath}/{args.dataset}-train.json', 'w') as f:
        # f.write(OmegaConf.to_yaml(augtext_bank))
        json.dump(train_json_data, f)
    with open(f'{args.savepath}/{args.dataset}-test.json', 'w') as f:
        json.dump(test_json_data, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DUCANH-UET")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='NCBI-disease', choices=['NCBI-disease', 'BC2GM', 'BC4CHEMD', 'BC5CDR-chem', 'BC5CDR-disease', 
                                                                                      'JNLPBA', 'linnaeus', 's800'])
    parser.add_argument('--datapath', type=str, default='../datasets/Compact-Biomedical/NER')
    parser.add_argument('--savepath', type=str, default='../datasets/slm4ner')
    parser.add_argument('--type', type=str, default='Disease')

    initial_time = time.time()
    args = parser.parse_args()
    args.tokenizerPath = "nlpie/distil-biobert" #model in  hugginghub
    args.datasetPath = f"{args.datapath}/{args.dataset}/"
    main(args)
    print("\n\n\t\t\t =====================The conversion took ", (time.time() -initial_time)/3600, " hours==============================")
