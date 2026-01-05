<img align="right" src="https://raw.githubusercontent.com/araujoeverton/ieee-ml-fraud-detection-ensemble/refs/heads/main/assets/project-cover.gif" width="1080"/> ...



<img align="right" src="https://raw.githubusercontent.com/araujoeverton/ieee-ml-fraud-detection-ensemble/cf1655fa9d788eee73c9130cb1160737e850170c/assets/python.svg" width="120"/>

# Usage Authorization
### 1. Save a copy and use the material in this repository for study purposes!
<a href="https://github.com/araujoeverton/ieee-ml-fraud-detection-ensemble/fork">
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

## 🔒 🔒 Security & Infrastructure Setup

To avoid 403 Access Denied errors and ensure production-grade security, this project uses a custom Google Cloud Project for DVC authentication.

### 1. Google Cloud Console Configuration

Before running the project, you must generate your own OAuth 2.0 credentials:

1. Go to the Google Cloud Console.

2. Create a new project (e.g., ```DVC-Fraud-Storage```).

3. Enable the ***Google Drive API***.

4. Configure the ***OAuth Consent Screen*** as "External" and add your email as a ***Test User.***

5. Create ***Credentials > OAuth 2.0 Client ID*** for "Desktop App".

6. Copy your ```Client ID``` and ```Client Secret```.

### 2. Environment Variables (.env)

1. Copy ```.env.example``` to a new file named ```.env```.

2. Fill in the following variables:

. ```GDRIVE_FOLDER_ID```: The ID from your Google Drive folder URL.

. ```GDRIVE_CLIENT_ID```: The ID generated in Google Cloud Console.

. ```GDRIVE_CLIENT_SECRET```: The Secret generated in Google Cloud Console.

### 3. Automated Setup

Once your ```.env``` is ready, run the main setup script to link DVC with your custom credentials:

```text
chmod +x setup_remote.sh
./setup_remote.sh
```

> [!IMPORTANT]
> Never commit your real `.env` file. Recruiters: this setup ensures the project is portable and secure for team collaboration.

## ⚙️ Setup & Installation
### 1. Environment & Dependencies

Ensure you are using Python 3.10+ (as seen in our PyCharm configuration). Install all necessary libraries:

```text
pip install -r requirements.txt
```


## 📦 Data Management with DVC

Since the IEEE-CIS dataset is too large for GitHub, I implemented **DVC (Data Version Control)**. This allows the repository to stay lightweight while maintaining full traceability of data versions.

***Storage Logic:*** Large files are stored as encrypted hashes on Google Drive to ensure version integrity.

***Syncing:*** Use ```dvc push``` to upload local changes and dvc pull to download data.

***Metadata:*** Only the ```.dvc``` pointer files are tracked by Git, keeping the repository lightweight.



### 🔄 Summary of Commands
| Command                          | Purpose |
|:---------------------------------| :--- |
| `./setup_remote.sh`              | One-time setup: links your `.env` secrets to DVC. |
| `dvc add input/archive_name.csv` | Tells DVC to track a large file. |
| `dvc push`                       | Uploads tracked files to Google Drive. |
| `dvc pull`                       | Downloads the data to a new machine. |

<div align="center">
  <p>
      <img src="https://img.shields.io/github/languages/count/alexklenio/DIO-dotnet-developer"/>
      <img src="https://img.shields.io/github/repo-size/alexklenio/DIO-dotnet-developer"/>
      <img src="https://img.shields.io/github/last-commit/alexklenio/DIO-dotnet-developer"/>
      <img src="https://img.shields.io/github/issues/alexklenio/DIO-dotnet-developer"/>
  </p> 
</div>


