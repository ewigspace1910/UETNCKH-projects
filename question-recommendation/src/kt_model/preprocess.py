# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import timedelta  
import logging
import os
import yaml 
from omegaconf import DictConfig, OmegaConf, open_dict

def filter4kt_df(df:pd.DataFrame):
    #record > 2020
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[df['timestamp']> np.datetime64("2020-05-30")]

    # Function to filter records within 0.2 hour range and get the last record
    def get_last_record(group):
        filtered_group = group[group['timestamp'] >= group['timestamp'].max() - timedelta(hours=0.2)]
        return filtered_group.tail(1)
    groups = df.groupby(['user_id', "item_id", "question_id"])
    new_df = pd.concat([get_last_record(group) for _, group in groups])
    new_df.reset_index(drop=True, inplace=True)

    new_df.sort_values(by='timestamp', ascending=True, inplace=True)  # Sort the DataFrame by 'Time' column

    return new_df


def merge_dfs(config, args):
    raw_folder = os.path.join(os.getcwd(), config.data.raw_folder)
    worksheet_scores=pd.read_csv(os.path.join( raw_folder,f'pals_performance_score-dbid-{args.dbid}.csv')).rename(columns={"created_at":"timestamp", 'topic_id':'item_id'})
    question_info=pd.read_csv(os.path.join( raw_folder,f'pals_questions-dbid-{args.dbid}.csv'), usecols=['question_id', "difficulty_level", "facility_index"])
    topic_info=pd.read_csv(os.path.join( raw_folder,f'pals_topics-dbid-{args.dbid}.csv'))
    worksheet_scores['percent'] = worksheet_scores['score'] / (worksheet_scores['fullmark'] + 1e-6)
    worksheet_scores['correct'] = worksheet_scores['percent'].apply(lambda x: 0 if x < 0.5 else 1)

    fake_df = topic_info.rename(columns={"topic_id":"item_id", 'substrand_id':'skill_id'})
    fake_df['user_id'] = [-1] * len(topic_info)
    fake_df['difficulty'] = fake_df['user_id'].apply(lambda x: np.random.choice([20, 40, 60, 66, 80, 99], p=[0.05, 0.1, 0.2, 0.4, 0.2, 0.05]))
    fake_df['correct'] = fake_df['user_id'].apply(lambda x: np.random.choice([1, 0], p=[0.8, 0.2]))
    fake_df['percent'] = fake_df['correct']
    print(fake_df)

    black_users = pd.read_csv(config.data.black_lists.user, usecols=['user_id','total'])
    question_info['difficulty_level'] = question_info['difficulty_level'].apply(lambda x: 99 if x=='hard' else 33 if x=='easy' else 66 )
    question_info['has_FIscore'] =  question_info['facility_index'].apply(lambda x :  1 if x > 0 else 0)
    question_info['difficulty_level'] = question_info['has_FIscore'] * question_info['facility_index'] * 100 + (1 - question_info['has_FIscore']) * question_info['difficulty_level']
    question_info.drop(['has_FIscore', 'facility_index'], axis=1)
    #drop na and join with user_student
    worksheet_scores = pd.merge(worksheet_scores, black_users, on='user_id', how='left')
    worksheet_scores = pd.merge(worksheet_scores, topic_info, left_on="item_id",right_on='topic_id', how='left')
    worksheet_scores.dropna(subset=['subject_id', "substrand_id"], inplace=True)
    worksheet_scores = worksheet_scores[worksheet_scores['total'].isnull()] #exclude demo-users
    worksheet_scores.drop(['total'], axis=1, inplace=True)
    worksheet_scores = pd.merge(worksheet_scores, question_info, on='question_id')
    worksheet_scores = filter4kt_df(worksheet_scores)
    
    
    worksheet_scores['difficulty'] = worksheet_scores['difficulty_level']
    worksheet_scores['skill_id'] =  worksheet_scores['substrand_id']

    # save
    new_df = worksheet_scores[['user_id', "item_id", "subject_id", "skill_id", "difficulty", 
                               "correct","percent"]]
    print(new_df.head())
    print("After processing, The filtered df has ",len(new_df), " row")

    # add fake data to cover all topics in db
    new_df = pd.concat([fake_df, new_df])
    print(new_df.head())
    print("After processing and adding fake noise, The filtered df has ",len(new_df), " row")
    #fillter
    new_df.to_csv(os.path.join( raw_folder, f"kt_merged_df-dbid-{args.dbid}.csv"), index=False)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Finetune Training")
    parser.add_argument('-s', '--dbid', type=str)
    args = parser.parse_args()
    cfgpath = os.path.join(os.getcwd(), "cfg/config.yml")
    # with open(cfgpath, "r") as config_file:
    #     config = yaml.load(config_file, Loader=yaml.FullLoader)
    config = OmegaConf.load(cfgpath)
    
    merge_dfs(config, args)
