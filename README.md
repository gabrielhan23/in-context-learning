# 182 Group Project YAY

## Setup
```
python -m venv .venv
source .venv/bin/activate
pip install torch tqdm matplotlib numpy pandas scikit-learn seaborn transformers wandb xgboost protobuf nbstripout
nbstripout --install
```

## Train: 
```
python src/train.py --config src/conf/toy.yaml
```