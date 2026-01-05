import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Encontra o caminho da raiz do projeto (onde o .env está)
base_path = Path(__file__).resolve().parent.parent
env_path = base_path / '.env'

# Carrega o .env apontando o caminho exato
load_dotenv(dotenv_path=env_path)

gdrive_id = os.getenv('GDRIVE_FOLDER_ID')

if gdrive_id:
    # dvc remote add -d (default) myremote
    cmd = f"dvc remote add -d myremote gdrive://{gdrive_id}"
    # Executa o comando
    subprocess.run(cmd, shell=True)
else:
    print("Variable GDRIVE_FOLDER_ID not found in .env")