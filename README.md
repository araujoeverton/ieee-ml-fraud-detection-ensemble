<img align="right" src="https://raw.githubusercontent.com/araujoeverton/ieee-ml-fraud-detection-ensemble/refs/heads/main/assets/project-cover.gif" width="1080"/> ...



<img align="right" src="https://raw.githubusercontent.com/araujoeverton/ieee-ml-fraud-detection-ensemble/cf1655fa9d788eee73c9130cb1160737e850170c/assets/python.svg" width="120"/>

# Autorização de Uso
### 1. Salve uma cópia e utilize o material deste repositório para estudo!
<a href="https://github.com//araujoeverton/XP_Bootcamp_Engenharia_de_Dados//fork">
    <img alt="Folk" title="Fork Button" src="https://shields.io/badge/-DAR%20FORK-red.svg?&style=for-the-badge&logo=github&logoColor=white"/></a>


## Project Details

This project implements a modular Machine Learning pipeline to detect fraudulent transactions using the IEEE-CIS dataset. It features a multi-model ensemble approach (LightGBM & XGBoost) with optimized memory management and categorical encoding.

## 📁 Project Structure

```text
projeto_fraude/
│
├── setup/
│   └── setup_remote.py      # Python logic for secure GDrive configuration
├── src/
│   ├── 01_preprocessing.py  # Data cleaning, Engineering & Encoding
│   ├── 02_train_lgbm.py     # LightGBM training with K-Fold
│   └── utils.py             # Memory reduction & shared helpers
├── input/                   # Raw CSV data (Tracked by DVC)
├── processed_data/          # Optimized .parquet files (Tracked by DVC)
├── models/                  # Serialized models and encoders
├── predictions/             # OOF and Test predictions for ensembling
├── .env.example             # Template for secure environment variables
├── requirements.txt         # Project dependencies
└── setup_remote.sh          # Root automation script for setup
```



## ⚙️ Setup & Installation
### 1. Environment & Dependencies

Ensure you are using Python 3.10+ (as seen in our PyCharm configuration). Install all necessary libraries:

```text
pip install -r requirements.txt
```

### 2. Secure Data Configuration (DVC)

We use DVC to manage large datasets externally. To configure your remote storage securely:

1. Copy `.env.example` to a new file named `.env`.

2. Fill in your `GDRIVE_FOLDER_ID` in the `.env` file.

3. Run the automated root script:

```text
chmod +x setup_remote.sh
./setup_remote.sh
```

<div align="center">
  <p>
      <img src="https://img.shields.io/github/languages/count/alexklenio/DIO-dotnet-developer"/>
      <img src="https://img.shields.io/github/repo-size/alexklenio/DIO-dotnet-developer"/>
      <img src="https://img.shields.io/github/last-commit/alexklenio/DIO-dotnet-developer"/>
      <img src="https://img.shields.io/github/issues/alexklenio/DIO-dotnet-developer"/>
  </p> 
</div>


