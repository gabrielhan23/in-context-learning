# 182 Group Project YAY

## Setup
```
python -m venv .venv
source .venv/bin/activate
pip install torch tqdm matplotlib numpy pandas scikit-learn seaborn transformers wandb xgboost protobuf nbstripout torch_print
nbstripout --install
```

## Train: 
```
python3 src/train.py --config src/conf/toy.yaml
```