# litgpt download TinyLlama/TinyLlama-1.1B-Chat-v1.0
MODELPATH=TinyLlama/TinyLlama-1.1B-Chat-v1.0
DATAFOLDER=../datasets/Compact-Biomedical/NER

MODELPATH=../output/tinyllama/BC2GM/final
python test_model.py -m $MODELPATH -d BC2GM -p $DATAFOLDER -t Gene/protein




echo ""
echo "JNLPBA"
MODELPATH=../output/tinyllama/JNLPBA/final
python test_model.py -m $MODELPATH -d JNLPBA -p $DATAFOLDER -t Gene/protein

