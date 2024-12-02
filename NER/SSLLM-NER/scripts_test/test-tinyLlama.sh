# litgpt download TinyLlama/TinyLlama-1.1B-Chat-v1.0
MODELPATH=TinyLlama/TinyLlama-1.1B-Chat-v1.0
DATAFOLDER=../datasets/Compact-Biomedical/NER

MODELPATH=../output/tinyllama/ncbi/final
python test_model.py -m $MODELPATH -d NCBI-disease -p $DATAFOLDER -t Disease




echo ""
echo "bc5cdr-disease"
MODELPATH=../output/tinyllama/bc5cdr-disease/final
python test_model.py -m $MODELPATH -d BC5CDR-disease -p $DATAFOLDER -t Disease

