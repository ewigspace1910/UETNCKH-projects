import requests
import json 
import pandas as pd
import json
import pymysql as mysql
import pandas as pd
import db_integration.query_rnd as QueryRnDBank
import db_integration.query as QueryBank
from const import db_config, URL_QUERY

def call_db(db_name, query, url_query=URL_QUERY, to_json=False):
    df = None
    try:
        if db_name == "original": db_name = 'pals'
        df = query2db(db_name, query)
        if df is None : raise Exception("Cannot query to DB with query : \n{}".format(query))
        if to_json: return {str(k):v for k, v in df.to_dict().items()}
        return df
    except Exception as e:
        print("\t===>[ERR] raise from db_integration.call_db",str(e))
        df = None
        try:
            response = requests.get(url_query, params={"db":db_name, "query":query}, headers={'Accept': 'application/json'})
            output = json.loads(response.text)
            
            if to_json: return output['body'] if 'body' in output.keys() else None
            return pd.DataFrame(output['body']) if 'body' in output.keys() else None
        except:
            print("\t===>[ERR] cannot connect to url_query")
            return None


def query2db(db, query):
    #run query
    if db == "pals" or db=="original" or db=="smartjen":    
        my_db = mysql.connect(host=db_config['pals']['hostname'],
                                        user=db_config['pals']['user'], 
                                        password=db_config['pals']['password'],
                                        db=db_config['pals']['schema_name'],
                                        port=db_config['pals']['port'],
                                        connect_timeout=10)
       

    elif db == 'alias4urDB': #add new DB source here
        my_db = mysql.connect(host=db_config['alias4urDB']['hostname'],
                        user=db_config['alias4urDB']['user'], 
                        password=db_config['alias4urDB']['password'],
                        db=db_config['alias4urDB']['schema_name'],
                        port=db_config['alias4urDB']['port'],
                        connect_timeout=30)
    
    else:            
       raise "body['db'] must be either ['smartjen', 'rnd', 'hwdb'] with smartjen is student data, rnd is rnd data"
    
    #convert data to dict
    if query.upper().strip().find("SELECT") == 0:
        results = pd.read_sql_query(query, my_db)
    else:
        conn = my_db.cursor()
        conn.execute(query)
        results = pd.DataFrame()
        my_db.commit()
    return results

