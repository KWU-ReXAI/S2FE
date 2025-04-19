### wandb를 해보아요.

import wandb
import os
from dotenv import load_dotenv
import subprocess
import pandas as pd

load_dotenv()

subprocess.run(f"python datapreprocessing.py --isall True", shell=True)
subprocess.run(f"python datapreprocessing.py --isall cluster", shell=True)
subprocess.run(f"python clustering.py", shell=True)
subprocess.run(f"python datapreprocessing.py --isall False", shell=True)
  
# 1. Sweep 설정 (Python dict)
sweep_config = {
	"method": "grid",  # grid, random, bayes 중 택1
	"metric": {
		"name": "CAGR",
		"goal": "maximize"
	},
	"parameters": {
		"lr_MLP": {
			"values": [0.0001, 0.001, 0.01]
		},
		"lr_anfis": {
			"values": [0.001, 0.01, 0.1]
		},
		"epochs_MLP": {
			"values": [100, 200, 300]
		},
		"epochs_anfis": {
			"values": [100, 200, 300]
		},
		"hidden": {
			"values": [128, 256]
		}
	}
}

def sweep():
	with wandb.init() as run:
		config = run.config
		# 포멧에 맞게 파일들을 수정해야 함.
		# 인자를 추가하기!
		
		subprocess.run(f"python train.py --testNum 5", shell=True)

		cagr = []
		sharpe_ratio = []
		mdd = []
		# trainNum별로 결과가 각각 저장돼있어서 다 읽어오는 부분분
		for trainNum in range(5):
			dir = f"./result/train_result_dir_{trainNum+1}/train_result_file_{trainNum+1}.csv" # 훈련 후 검증결과과 저장파일
			df = pd.read_csv(dir, index_col=0, header=0)

			cagr.append(df["Average"][0])
			sharpe_ratio.append(df["Average"][1])
			mdd.append(df["Average"][2])

		# 읽어오 결과로 평균내기기
		avg_cagr = sum(cagr) / len(cagr)
		avg_sharpe_ratio = sum(sharpe_ratio) / len(sharpe_ratio)
		avg_mdd = sum(mdd) / len(mdd)

		wandb.log({"CAGR": avg_cagr, "SHARPE RATIO": avg_sharpe_ratio, "MDD": avg_mdd})

# wandb api key 입력하기
wandb.login(key=os.getenv("WANDB_API_KEY"))
sweep_id = wandb.sweep(sweep_config, project="s3ce_hyper_tuning")
wandb.agent(sweep_id, function=sweep, count=30)