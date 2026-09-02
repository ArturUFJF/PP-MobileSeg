import os
import shutil
import subprocess
import sys

# Quantidade de splits do seu Cross-Validation
K_SPLITS = 5
CAMINHO_CONFIG = "configs/pp_mobileseg/pp_mobileseg_base_lsidbeans.yml"
DATASET_ROOT = "data/lsidbeans"
CV_ROOT = os.path.join(DATASET_ROOT, "cv_1")
ROOT = os.path.dirname(os.path.abspath(__file__))

for split in range(1, K_SPLITS + 1):
    print(f"\n{'='*40}")
    print(f" INICIANDO TREINAMENTO DO SPLIT {split}/{K_SPLITS}")
    print(f"{'='*40}\n")
    
    # 1. Copia as listas do fold atual para os caminhos lidos pelo YAML.
    pasta_fold = os.path.join(CV_ROOT, f"split_{split}")
    shutil.copyfile(os.path.join(pasta_fold, "train.txt"),
                    os.path.join(DATASET_ROOT, "train.txt"))
    shutil.copyfile(os.path.join(pasta_fold, "val.txt"),
                    os.path.join(DATASET_ROOT, "val.txt"))
    
    # 2. Define o diretório de saída para não sobrescrever os pesos do split anterior
    diretorio_saida = os.path.join("output", "pp_mobileseg_lsidbeans",
                                   f"L2Loss_cv1_split{split}")
    
    # 3. Monta o comando de treinamento nativo do PaddleSeg
    comando = [
        sys.executable,
        "tools/train.py",
        "--config",
        CAMINHO_CONFIG,
        "--save_dir",
        diretorio_saida,
        "--do_eval",
        "--save_interval",
        "500",
        "--seed",
        "1337",
    ]
    
    # 4. Executa o treinamento (pausa o loop de código até o comando terminar)
    subprocess.run(comando, cwd=ROOT, check=True)
    
print("\n[SUCESSO] Treinamento de todos os 5 splits finalizado!")