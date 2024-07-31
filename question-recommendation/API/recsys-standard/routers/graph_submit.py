from fastapi import APIRouter, File, UploadFile,  Form
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
import os
import pandas as pd
import random
import networkx as nx


from const import TMPROOT
from utils import call_db, submit_to_link_table

router = APIRouter(
    prefix="/submit",
    tags=['[dev-branch] Upload doc api ']
)

@router.get('/graph', response_class=HTMLResponse)
def get_ward(): 
    tb = call_db('smartjen', "SELECT id, name FROM sj_subject")
    choices = ''
    for _, id, name in tb.itertuples():
        choices += f'<option value={id}>{name}</option>\n'

    with open("./static/graphsubmit.html", 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip().startswith('<select id="subjectIDInput" required>'):
                lines[i] = line.split("|")[0] + choices + line.split("|")[1]
                break
        html_content = " ".join(lines)
    return HTMLResponse(content=html_content, status_code=200)

@router.post('/graph', summary="The DF should be in format head_id| head_name| rel | tail_id| tail_name")
async def submit_graphfile(subject_id:int=Form(...), docfile: UploadFile= File(...), is_test:bool=Form(...), show_requireRel:bool=Form(...), show_relevantRel:bool=Form(...)):
    try:
        filename = docfile.filename
        contents = await docfile.read()

        _, file_extension = os.path.splitext(filename)
        tmproot = TMPROOT if os.path.exists(TMPROOT) else "." 
        saved_path = os.path.join(tmproot, filename)
        with open(saved_path, "wb") as f: f.write(contents)

        if file_extension.lower() == ".csv": df = pd.read_csv(saved_path)
        elif file_extension.lower() == ".xlsx": df = pd.read_excel(saved_path)
        else: raise  'Unsupported file format'
        df_cp = df.copy()
        df_cp['head_id'] = df['tail_id']
        df_cp['tail_id'] = df['head_id']
        df_cp['head'] = df['tail']
        df_cp['tail'] = df['head']
        df_cp['rel'] = df['rel'].apply(lambda x: "reverse of require" if x.strip().lower()=='require' else 'require' if x.strip().lower() == 'is required by' else 'relevant' if x.strip().lower() == 'relevant' else "")
        df['rel']   = df['rel'].apply(lambda x : "reverse of require" if x == 'is required by' else x)
        dfmerge = pd.concat([df_cp, df]).drop_duplicates(subset=['head_id', 'tail_id'])
        
        dfmerge['check'] = dfmerge['rel'].apply(lambda x : x in ['require', 'relevant', 'is required by'])
        dfmerge = dfmerge[dfmerge['check']][['head', 'rel', 'tail', 'head_id', 'tail_id']]
        topic_bank = call_db("smartjen", f"SELECT  id, name, substrand_id  FROM sj_categories WHERE subject_type={subject_id}")
        topic2id = {item[2]:item[1] for item in topic_bank.itertuples()}
        topic2substrand=topic_bank.set_index("id")['substrand_id'].to_dict()

        dfmerge['head_substrand'] = dfmerge['head_id'].apply(lambda x: topic2substrand[x])
        dfmerge['tail_substrand'] = dfmerge['tail_id'].apply(lambda x: topic2substrand[x])
        dfmerge = dfmerge.sort_values(by=['head_id', 'tail_id'])
        dfmerge.reset_index(drop=True, inplace=True)

        #convert relationship "relevant-require" to 'revese_of_require - require'
        extra_dfmerge = pd.DataFrame()
        for i in range(len(dfmerge)):
            _, rel, _, head_id, tail_id, _, _ = dfmerge.iloc[i]
            indices = dfmerge[(dfmerge['head_id'] == tail_id) & (dfmerge['tail_id'] == head_id)].index
            if len(indices)==0 or int(indices.values) >= len(dfmerge) : continue
            rel_ = dfmerge.iloc[int(indices[0]), 1]
            if rel == 'require':
                if rel_ == "relevant":
                    dfmerge.iloc[int(indices.values), 1] = "reverse of require"
                elif rel_ == 'require':
                    tmp = dfmerge.iloc[[i]]
                    tmp['rel'] = 'reverse of require'
                    extra_dfmerge = pd.concat([extra_dfmerge, tmp])
            elif rel == 'relevant' and rel_ == "require":
                    dfmerge.iloc[i, 1] = "reverse of require"
        dfmerge = pd.concat([dfmerge, extra_dfmerge])
        dfmerge.reset_index(drop=True, inplace=True)

        ######
        # weight relationship edges
        dfmerge['rel_weight'] = dfmerge['rel'].apply(lambda x: 1 if x=='relevant' else 2)
        query = """
            SELECT topic_id, sum(num) as num FROM (
                SELECT topic_id , count(*) as num FROM sj_questions WHERE topic_id > 0 AND subject_type=2 GROUP BY topic_id
                UNION ALL
                SELECT topic_id2, count(*) as num FROM sj_questions WHERE topic_id2 > 0 AND subject_type=2 GROUP BY topic_id2
                UNION ALL
                SELECT topic_id3, count(*) as num FROM sj_questions WHERE topic_id3 > 0 AND subject_type=2 GROUP BY topic_id3
                UNION ALL
                SELECT topic_id4, count(*) as num FROM sj_questions WHERE topic_id4 > 0 AND subject_type=2 GROUP BY topic_id4
            ) as t
            GROUP BY topic_id
            """
        num_question_in_topic = call_db('smartjen',query)
        num_question_in_topic.set_index(['topic_id'], inplace=True)
        query = """
            SELECT t1, t2, sum(num) as num FROM (
                SELECT topic_id as t1, topic_id2 as t2, count(*) as num FROM sj_questions WHERE topic_id > 0 AND topic_id2 > 0 AND subject_type=2 GROUP BY topic_id, topic_id2
                UNION ALL
                SELECT topic_id, topic_id3, count(*) as num FROM sj_questions WHERE topic_id > 0 AND topic_id3 > 0 AND subject_type=2 GROUP BY topic_id, topic_id3
                UNION ALL
                SELECT topic_id, topic_id4, count(*) as num FROM sj_questions WHERE topic_id > 0 AND topic_id4 > 0 AND subject_type=2 GROUP BY topic_id, topic_id4

                UNION ALL
                SELECT topic_id2, topic_id,  count(*) as num FROM sj_questions WHERE topic_id > 0 AND topic_id2 > 0 AND subject_type=2 GROUP BY topic_id2, topic_id
                UNION ALL
                SELECT topic_id2, topic_id3, count(*) as num FROM sj_questions WHERE topic_id2 > 0 AND topic_id3 > 0 AND subject_type=2 GROUP BY topic_id2, topic_id3
                UNION ALL
                SELECT topic_id2, topic_id4, count(*) as num FROM sj_questions WHERE topic_id2 > 0 AND topic_id4 > 0 AND subject_type=2 GROUP BY topic_id2, topic_id4

                UNION ALL
                SELECT topic_id3, topic_id,  count(*) as num FROM sj_questions WHERE topic_id > 0 AND topic_id3 > 0 AND subject_type=2 GROUP BY topic_id3, topic_id
                UNION ALL
                SELECT topic_id3, topic_id2, count(*) as num FROM sj_questions WHERE topic_id2 > 0 AND topic_id3 > 0 AND subject_type=2 GROUP BY topic_id3, topic_id2
                UNION ALL
                SELECT topic_id3, topic_id4, count(*) as num FROM sj_questions WHERE topic_id3 > 0 AND topic_id4 > 0 AND subject_type=2 GROUP BY topic_id3, topic_id4

                UNION ALL
                SELECT topic_id4, topic_id,  count(*) as num FROM sj_questions WHERE topic_id > 0 AND topic_id4 > 0 AND subject_type=2 GROUP BY topic_id4, topic_id
                UNION ALL
                SELECT topic_id4, topic_id2, count(*) as num FROM sj_questions WHERE topic_id2 > 0 AND topic_id4 > 0 AND subject_type=2 GROUP BY topic_id4, topic_id2
                UNION ALL
                SELECT topic_id4, topic_id3, count(*) as num FROM sj_questions WHERE topic_id3 > 0 AND topic_id4 > 0 AND subject_type=2 GROUP BY topic_id4, topic_id3
            ) as t
            GROUP BY t1, t2
            """
        topic_pair = call_db('smartjen',query)
        for _, t1, t2, shared_num in topic_pair.itertuples():
            weight = shared_num / num_question_in_topic.loc[t1, 'num']
            head_id, tail_id = t1, t2
            indices = dfmerge[(dfmerge['head_id'] == head_id) & (dfmerge['tail_id'] == tail_id)].index
            for i in indices:
                dfmerge.iloc[i, -1] += weight

        topic_graph = nx.from_pandas_edgelist(dfmerge, 'head_id', 'tail_id',
                            edge_attr=["rel", 'rel_weight', 'head_substrand',  'tail_substrand'],
                            create_using=nx.MultiDiGraph())

        # make graph for substrand        


        # save graph and push into S3
        if not is_test:
            save_name = f"topicG_subject_{subject_id}"
            nx.write_gml(topic_graph, os.path.join(tmproot, save_name+'.gml'))
            # submit_to_link_table(os.path.join(tmproot, save_name+'.gml'), save_name, 'topic-graph')

        #######################
        # plot
        substrand2id = {}
        nodes = []
        edges = []
        group = 0
        check_id = set()
        dd = dfmerge[~(dfmerge['rel'] == 'reverse of require')]
        if not show_relevantRel : dd = dfmerge[~(dfmerge['rel'] == 'relevant')]
        if not show_requireRel : dd = dfmerge[~(dfmerge['rel'] == 'require')]
        
        dd['f1'] = dd['head_id'] + dd['tail_id']
        dd['f2'] = abs(dd['head_id'] - dd['tail_id'])
        dd['f3'] = (dd['f1'] + dd['f2'])/2 * 1e6 +  (dd['f1'] - dd['f2'])  #remove pair if they have same heads
        dd.drop_duplicates(subset=['f3'], inplace=True)
        for item in dd.itertuples():
            #Pandas(Index=0, head='Modal Verb', rel='require', tail='Verbs of Being/Linking Verbs ', head_id=75, tail_id=162, head_substrand=86, tail_substrand=86, rel_weight=2, f1=237, f2=87, f3=162000150.0)
            ss_head = item[6]
            ss_tail = item[7]
            if not ss_head in substrand2id.keys(): 
                substrand2id[ss_head] = [generate_random_hex_color(), group]
                group += 1
            if not ss_tail in substrand2id.keys(): 
                substrand2id[ss_tail] = [generate_random_hex_color(), group]
                group += 1
            node1 = {"color": substrand2id[ss_head][0], 
                    "group": substrand2id[ss_head][1], "id": item[4], 
                    "label": item[1], "shape": "dot", 
                    "size": 10, "title": item[1]}            
            node2 = {"color": substrand2id[ss_tail][0], 
                    "group": substrand2id[ss_tail][1], "id": item[5], 
                    "label": item[3], "shape": "dot", 
                    "size": 10, "title": item[3]}

            if item[4] not in check_id : nodes.append(node1)
            if item[5] not in check_id : nodes.append(node2)
            check_id =  check_id | {item[4], item[5]}
            

            edge = {"from": item[4], "to": item[5], "width": item[-4], 'label':item[2]}
            edges.append(edge)
                   
        return JSONResponse(content={'complete': True, 
                                     'data':{'nodes':nodes, 'edges':edges}})
    except Exception as e:
        error_msg = f"Error processing CSV file: {str(e)}"
        print(error_msg)
        return JSONResponse(content={'complete': False, 'error': error_msg})


#linhtinh
def generate_random_hex_color():
    color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    return color
