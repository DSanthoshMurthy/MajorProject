## StreamlitApp

### Setup

```bash
cd /Users/dsanthoshmurthy/Documents/StreamlitApp
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### Scrape and preprocess data for FinBERT

This step reads raw URLs from `data/raw_data.csv`, scrapes article text, and writes a cleaned dataset to `data/finbert_training_data.csv`.

```bash
cd /Users/dsanthoshmurthy/Documents/StreamlitApp
source .venv/bin/activate
python scrape_and_preprocess.py
```

### Train the FinBERT sentiment model

This step fine-tunes `ProsusAI/finbert` using `data/finbert_training_data.csv` and saves the model into the `model/` directory.

```bash
cd /Users/dsanthoshmurthy/Documents/StreamlitApp
source .venv/bin/activate
python train_finbert.py
```

### Run the Streamlit app

```bash
cd /Users/dsanthoshmurthy/Documents/StreamlitApp
source .venv/bin/activate
streamlit run app.py
```
