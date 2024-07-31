import pymysql as mysql

import os
import pandas as pd
import yaml
import os 
from omegaconf import DictConfig, OmegaConf, open_dict


def pull_raw_data(cfgpath, dbpath, args):
    db_config = OmegaConf.load(dbpath)
    config = OmegaConf.load(cfgpath)

    folder = config.data.raw_folder
    if not os.path.exists(folder): os.makedirs(folder)

    my_db = mysql.connect(host=db_config.indb.hostname,
                                    user=db_config.indb.user, 
                                    password=db_config.indb.password,
                                    db=db_config.indb.schema_name,
                                    port=db_config.indb.port)
    db_schema = db_config.indb.schema_name
    

    # save tables
    for table in db_config.tables.keys():
        columns = ",".join(db_config.tables[table])
        print(f"Select table {table} with cols : {columns}")
        filename = os.path.join(folder, table + f'-dbid-{args.dbid}.csv')
        abs_filename = os.path.abspath(filename)
        query = f"select {columns} from {db_schema}.{table} WHERE pals_dbid={args.dbid}"
        print(query)
        results = pd.read_sql_query(query, my_db)
        print("\n\t----", len(results), " rows\n")
        results.to_csv(abs_filename, index=False)
        print(f"Pulling database table {table}, saving to {abs_filename}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Finetune Training")
    parser.add_argument('-s', '--dbid', type=str)
    args = parser.parse_args()
    dir_path = os.getcwd()
    dbpath = os.path.join(dir_path, "cfg/db.yml")
    cfgpath = os.path.join(dir_path, "cfg/config.yml")
    pull_raw_data(cfgpath, dbpath, args)