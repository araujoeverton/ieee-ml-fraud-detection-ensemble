<img align="right" src="https://raw.githubusercontent.com/araujoeverton/ieee-ml-fraud-detection-ensemble/refs/heads/main/assets/project-cover.gif" width="1080"/> ...



<img align="right" src="https://raw.githubusercontent.com/araujoeverton/ieee-ml-fraud-detection-ensemble/cf1655fa9d788eee73c9130cb1160737e850170c/assets/python.svg" width="120"/>

# Usage Authorization
### 1. Save a copy and use the material in this repository for study purposes!
<a href="https://github.com//araujoeverton/ieee-ml-fraud-detection-ensemble//fork">
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

## 🔒 Security & Configuration

This project uses **environment variables** to manage sensitive information, such as Google Drive Folder IDs. This prevents hardcoding credentials and ensures the project follows security best practices.

### The `.env.example` File
I have provided a `.env.example` file to guide the setup:
- **Duplicate** `.env.example` and rename it to `.env`.
- **Fill** in your private `GDRIVE_FOLDER_ID`.
- **Note:** The `.env` file is explicitly ignored by Git (see `.gitignore`) to prevent accidental leaks.

> [!IMPORTANT]
> Never commit your real `.env` file. Recruiters: this setup ensures the project is portable and secure for team collaboration.

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

## 📦 Data Management with DVC

Since the IEEE-CIS dataset is too large for GitHub, I implemented **DVC (Data Version Control)**. This allows the repository to stay lightweight while maintaining full traceability of data versions.

### 1. The Setup Flow (`setup_remote.sh`)
The configuration is automated to ensure security and ease of use:
- **Security**: The script reads your `GDRIVE_FOLDER_ID` from a local `.env` file (not tracked by Git).
- **Automation**: By running `./setup_remote.sh`, the system activates the virtual environment, installs dependencies, and links DVC to your private Cloud Storage.

### 2. The Data Push (`dvc push`)
Once the remote storage is configured, you can sync your local data with the cloud:
- **How it works**: When you run `dvc add <file>`, DVC creates a lightweight `.dvc` pointer file.
- **Pushing**: Running `dvc push` uploads the actual heavy files (CSVs, Parquets, Models) to Google Drive.
- **Collaboration**: Anyone else with access can simply run `dvc pull` to get the exact same data versions without manually downloading from Kaggle.



### 🔄 Summary of Commands
| Command | Purpose |
| :--- | :--- |
| `./setup_remote.sh` | One-time setup: links your `.env` secrets to DVC. |
| `dvc add input/data.csv` | Tells DVC to track a large file. |
| `dvc push` | Uploads tracked files to Google Drive. |
| `dvc pull` | Downloads the data to a new machine. |

<div align="center">
  <p>
      <img src="https://img.shields.io/github/languages/count/alexklenio/DIO-dotnet-developer"/>
      <img src="https://img.shields.io/github/repo-size/alexklenio/DIO-dotnet-developer"/>
      <img src="https://img.shields.io/github/last-commit/alexklenio/DIO-dotnet-developer"/>
      <img src="https://img.shields.io/github/issues/alexklenio/DIO-dotnet-developer"/>
  </p> 
</div>


