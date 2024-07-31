import boto3
from botocore import UNSIGNED
from botocore.client import Config as AWSConfig
import networkx as nx
import const
from const import *
from utils import *
from db_integration import call_db, QueryBank, QueryRnDBank

def get_graph(db_id, subject_id, root="/tmp"):
    try:
        path_model = os.path.join(root, f"topicG_subject_{subject_id}-dbid-{db_id}.gml")
        if not os.path.exists(path_model):
            df = call_db("pals", QueryRnDBank.get_model_config_link(db_id=db_id, category='KG', model_name=f'bigKG_subject_{subject_id}') )
            if len(df) == 0: return None
            model_link = df[df['extention'] == "gml"]['link'][0].split(".com/")[-1]
            BUCKET_NAME = 'pdf-digitalization' # replace with your bucket name
            s3 = boto3.client('s3', config=AWSConfig(signature_version=UNSIGNED))
            with open(path_model, "wb") as data:
                s3.download_fileobj(BUCKET_NAME, model_link, data)
        
        G = nx.read_gml(path_model)
        return G
    except Exception as e:
        LOGGER.exception(e, "[ERR] in logic_graph.py | inputs={}".format(subject_id))
        return None

def extract_neighbor(db_id:int, subject_id, topic_id:int,  relation_types=["relevant", 'require', 'reverse of require'], min_weight=0, student_level=None, **kwargs):
    edges_df=None
    try:
        #Get edge_df from Huawei GES
        # edges_df = get_graph_from_GES(subject_id, topic_id, relation_types,min_weight)
        # if edges_df is None:
        graph = get_graph(db_id, subject_id, root=TMPROOT)
        # Extract edges based on conditions
        # edges_to_extract = [(int(v), edge_attrs.get('rel_weight'), int(edge_attrs.get('head_substrand') == edge_attrs.get('tail_substrand')) ) for u, v, edge_attrs in graph.edges(data=True)
        #                     if int(u) == topic_id and edge_attrs.get('rel') in relation_types and edge_attrs.get('rel_weight') > min_weight]
        edges_to_extract = [ ]
        for u, v, edge_attrs in graph.edges(data=True):
            tid_head = int(u.split("-")[1]) # in graph, topic is stored in format as 't-122-l3' with 122 is topic_id, l3 is grade level 3
            head_level = int(u.split("-")[2][1:])
            tid_tail = int(v.split("-")[1])
            tail_level = int(v.split("-")[2][1:])
            rel = edge_attrs.get('rel')
            try: rel_weight = float(edge_attrs.get('rel_weight'))  
            except : rel_weight=0
            if tid_head == topic_id and rel in relation_types and rel_weight > min_weight and (student_level is None or (student_level == head_level)):
                if student_level is None or (student_level == tail_level):
                    edges_to_extract += [(tid_tail, edge_attrs.get('rel_weight'), 
                                          int(edge_attrs.get('head_substrand') == edge_attrs.get('tail_substrand')))]
        # Create a DataFrame from the extracted edges
        edges_df = pd.DataFrame(edges_to_extract, columns=['topic_id', 'weight', 'issamegroup'])
    except Exception as e:
        LOGGER.exception(e, "[ERR] in logic_graph.extract_neighbor | inputs={}".format([subject_id, topic_id, relation_types]))
        edges_df = None
    
    return edges_df



def topic_filters(df, student_id, subject_id, student_level, extra_levels=[], sort_type=1, from_db:int=1):
    """
    The function `topic_filters` takes in a dataframe, student ID, subject ID, student level, extra
    levels, and sort type, and returns a filtered and sorted dataframe based on certain criteria.

    :param df: The input dataframe containing the topic_id column
    :param student_id: The student_id parameter is the ID of the student for whom we want to filter the topics
    :param subject_id: The `subject_id` parameter represents the ID of the subject for which you want to filter topics
    :param student_level: The student_level parameter represents the level of the student. It is used to filter topics based on the student's level
    :param extra_levels: The `extra_levels` parameter is a list of additional student levels that you
    want to consider when filtering topics. These levels are in addition to the `student_level`
    parameter, which represents the primary student level. By including extra levels, you can filter topics that are relevant to multiple student levels
    :param sort_type: The `sort_type` parameter determines how the topics should be sorted. There are
    three options:, defaults to 1 (optional)
    :return: a DataFrame with filtered topics based on certain criteria.
    """
    try:
        seen_topics = call_db("pals", QueryBank.analyze_top_weak_topic(db_id=from_db, student_id=student_id, subject_id=subject_id))
        df = pd.merge(df, seen_topics, on="topic_id", how='left')
        df['psuedo_point'] = df['psuedo_point'].apply(lambda x: 0 if x < 5 else  50 if x < 50 else 100 if x < 100 else 500 if x < 500 else 1000)
        df['weight'] = df['weight'].apply(lambda x: round(x, 2))

        df['issamegroup'] = df['issamegroup'].apply(lambda x: 0.04 if x else 0)
        df['weight'] = df['weight'] + df['issamegroup']

        # 1. select topic in same student level/extra levels
        topic_levels = call_db("pals", QueryBank.get_available_topic_ids(db_id=from_db, subject_id=subject_id, 
                                                                    student_level=student_level, extra_levels=extra_levels))
        df = pd.merge(df, topic_levels, on="topic_id")


        # 2. sort topics
        if sort_type == 1: #pioritize to explore new topic
            df.sort_values(by=[ 'psuedo_point', 'weight'], ascending=[True, False ], inplace=True)
        elif sort_type==2: #pioritize to explore close topic
            df.sort_values(by=[ 'psuedo_point', 'weight'], ascending=[True, False ], inplace=True)
        
        elif sort_type==3: #pioritize to practice prequisited topic
            df['isrequired1'] = df['weight'].apply(lambda x: x >= 2)
            df['isrequired2'] = df['psuedo_point'].apply(lambda x: x < 50)
            df['isrequired']  = df['isrequired1'] * df['isrequired2']
            df.sort_values(by=[ 'isrequired', 'psuedo_point' , 'weight'], ascending=[ False, True,False ], inplace=True)
        # print(df)
        return df

    except Exception as e:
        LOGGER.exception(e, "[ERR] in logic_graph.topic_fillters | inputs={}".format([subject_id, student_id, subject_id, student_level, extra_levels]))
        return None




def find_relevant_topic(current_topic_id, subject_id, student_id, student_level, 
                        rel_types=["relevant",  'revese of require'], extra_levels=[], 
                        focus_history=False, sort_type=1, from_db:int=1,**kwargs):
    try:
        if subject_id not in const.KGsupported_subjects: return []
        edges_df = extract_neighbor(from_db, subject_id, current_topic_id, relation_types=rel_types, min_weight=0, student_level=student_level)
        edges_df = topic_filters(edges_df, student_id, subject_id, student_level, extra_levels, sort_type=sort_type,from_db=from_db)
        # prioritize topic most related to current question student answerred wrongly
        # failure_topics = call_db("pals", QueryBank.analyze_top_failure_topic(student_id, subject_id))
        # if not focus_history or len(failure_topics)==0: return edges_df['topic_id'].tolist()[:5]
        # mix_df = pd.merge(edges_df, failure_topics, on='topic_id', how='outer').fillna(0)
        mix_df =edges_df
        mix_df['weight'] = mix_df['weight'] + mix_df['failures']/(1e-9+mix_df['falures']) * 0.005
        mix_df.sort_values(by=[ 'weight', 'psuedo_point'], ascending=[False, True], inplace=True)
        return mix_df['topic_id'].tolist()[:5]
    except Exception as e:
        LOGGER.exception(e, "[ERR] error in find relevant_topic with info:", current_topic_id, subject_id, student_id, student_level, rel_types, focus_history)
        return []



