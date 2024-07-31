DATAPATH=../datasets/Compact-Biomedical/NER
SAVEPATH=../datasets/slm4ner
python datasets/convert_data_json.py -d NCBI-disease --type Disease --datapath $DATAPATH --savepath $SAVEPATH
python datasets/convert_data_json.py -d BC5CDR-disease --type Disease --datapath $DATAPATH --savepath $SAVEPATH

python datasets/convert_data_json.py -d BC2GM --type Gene/protein --datapath $DATAPATH --savepath $SAVEPATH
python datasets/convert_data_json.py -d JNLPBA --type Gene/protein --datapath $DATAPATH --savepath $SAVEPATH

python datasets/convert_data_json.py -d BC4CHEMD --type Drug/chem --datapath $DATAPATH --savepath $SAVEPATH
python datasets/convert_data_json.py -d BC5CDR-chem --type Drug/chem --datapath $DATAPATH --savepath $SAVEPATH

python datasets/convert_data_json.py -d linnaeus --type Species --datapath $DATAPATH --savepath $SAVEPATH
python datasets/convert_data_json.py -d s800 --type  Species --datapath $DATAPATH --savepath $SAVEPATH
