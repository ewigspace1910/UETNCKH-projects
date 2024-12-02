# litgpt download TinyLlama/TinyLlama-1.1B-Chat-v1.0
MODELPATH=TinyLlama/TinyLlama-1.1B-Chat-v1.0
DATAFOLDER=../datasets/Compact-Biomedical/NER

MODELPATH=../output/tinyllama/BC4CHEMD/final
python test_model.py -m $MODELPATH -d BC4CHEMD -p $DATAFOLDER -t Drug/chem




echo ""
echo "BC5CDR-chem"
MODELPATH=../output/tinyllama/BC5CDR-chem/final
python test_model.py -m $MODELPATH -d BC5CDR-chem -p $DATAFOLDER -t Drug/chem

