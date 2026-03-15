# Physics Guided Probabilistic Transformers for Amortized Inference in Quantitative Myocardial Perfusions

https://drive.google.com/file/d/1ZoPWgzED4ce58MVmMdhxc7krN6CZjfvL/view?usp=sharing

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
