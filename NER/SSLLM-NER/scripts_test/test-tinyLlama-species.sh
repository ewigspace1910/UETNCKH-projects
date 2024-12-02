# litgpt download TinyLlama/TinyLlama-1.1B-Chat-v1.0
MODELPATH=TinyLlama/TinyLlama-1.1B-Chat-v1.0
DATAFOLDER=../datasets/Compact-Biomedical/NER

MODELPATH=../output/tinyllama/s800/final
python test_model.py -m $MODELPATH -d s800 -p $DATAFOLDER -t Species




echo ""
echo "linnaeus"
MODELPATH=../output/tinyllama/linnaeus/final
python test_model.py -m $MODELPATH -d linnaeus -p $DATAFOLDER -t Species

