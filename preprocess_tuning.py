### wandb를 해보아요.

import wandb
import os
from dotenv import load_dotenv
import subprocess
import pandas as pd

load_dotenv()

# 1. Sweep 설정 (Python dict)
sweep_config = {
	"method": "grid",  # grid, random, bayes 중 택1
	"metric": {
		"name": "CAGR",
		"goal": "maximize"
	},
	"parameters": {
		"n_features_t" : {"values" : [4,5,6,7,8]}
	}
}

def sweep():
	with wandb.init() as run:
		config = run.config
		# 포멧에 맞게 파일들을 수정해야 함.
		# 인자를 추가하기!
		subprocess.run(f"python datapreprocessing.py --use_all True --isall True --n_features_t {config.n_features_t}", shell=True)
		subprocess.run(f"python datapreprocessing.py --use_all True --isall cluster --n_features_t {config.n_features_t}", shell=True)
		subprocess.run(f"python clustering.py", shell=True)
		subprocess.run(f"python datapreprocessing.py --use_all True --isall False --n_features_t {config.n_features_t}", shell=True)
		subprocess.run(f"python train.py --testNum 20", shell=True)
		subprocess.run(f"python test.py --testNum 20", shell=True)
  
		# testNum별로 결과가 각각 저장돼있어서 다 읽어오는 부분
		dir = f"./result/result_S3CE_SectorAll/test_result_dir/test_result_file.csv" # 훈련 후 검증결과과 저장파일
		df = pd.read_csv(dir, header=[0, 1], index_col=0)
		df_avg = df["Average"].mean(axis=1)
  
		wandb.log({"CAGR": df_avg['CAGR'], "SHARPE RATIO": df_avg['Sharpe Ratio'], "MDD": df_avg['MDD']})

# wandb api key 입력하기
wandb.login(key=os.getenv("WANDB_API_KEY"))
sweep_id = wandb.sweep(sweep_config, project="S3CE_n_features_t_experiment")
wandb.agent(sweep_id, function=sweep, count=30)
