# MeLi Challenge: Item Condition Prediction  

## 📂 Repository Content  
This repository contains:  
1. **EDA** → Variable analysis (`challenge/eda/`). Includes a notebook with per-variable analysis (approximate).  
2. **Dataset preparation** → Notebooks and scripts to build train/test sets (`challenge/dataset/`).  
3. **Model training** → Training notebook, best parameters, and results (`challenge/models/`).  
4. **Research** → Additional experiments with alternative models, encoders, and tests (`challenge/research/`).  

---

## ⚙️ Reproducibility  

To set up the environment and run the project from the repository root:

```bash
pyenv install 3.12
pip install poetry
python -m venv .venv
source .venv/bin/activate
poetry install
```
If you don't have pyenv installed just follow the steps [here](https://github.com/pyenv/pyenv?tab=readme-ov-file#a-getting-pyenv) and remember to install the dependencies first, more info [here](https://github.com/pyenv/pyenv/wiki#suggested-build-environment).