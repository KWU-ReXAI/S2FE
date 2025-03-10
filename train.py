import argparse
import pandas as pd
import torch
import warnings

from datamanager import DataManager
from model import MyModel
from tqdm import tqdm
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


warnings.filterwarnings(action='ignore')

parser = argparse.ArgumentParser() # 입력 받을 하이퍼파라미터 설정

parser.add_argument('--features_n',type=int,nargs='?',default=6) # 사용할 재무지표(feature) 개수
parser.add_argument('--valid_sector_k',type=int,nargs='?',default=2) # 검증 단계에서 선택할 상위 섹터 개수
parser.add_argument('--valid_stock_k',type=int,nargs='?',default=5) # 검증 단계에서 선택할 상위 주식 개수
parser.add_argument('--each_sector_stock_k',type=int,nargs='?',default=5) # 각 섹터에서 선택할 주식 개수
parser.add_argument('--final_stock_k',type=int,nargs='?',default=10) # 최종적으로 선택할 주식 개수
parser.add_argument('--result_name',type=str,nargs='?',default="train_result_model") # 결과 파일명
parser.add_argument('--dir_name',type=str,nargs='?',default="train_result_dir") # 결과 디렉토리 명
parser.add_argument('--aggregate',type=str,nargs='?',default="inter") # 교집합 또는 평균 방식을 선택
parser.add_argument('--use_all',type=str,nargs="?",default="SectorAll") # 모델을 평가하는 방식 설정
parser.add_argument('--ensemble',type=str,nargs="?",default="S3CE") # 사용할 앙상블 방법
parser.add_argument('--clustering',action="store_true",default=True) # 클러스터링 여부
parser.add_argument('--testNum',type=int,nargs='?',default=0) # 클러스터링 여부

args = parser.parse_args()
DM = DataManager(args.features_n)
result = {}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dir = f"./result/{args.dir_name}_{args.testNum}" # 결과를 저장할 디렉토리 생성
if os.path.isdir(dir) == False:
    os.mkdir(dir)


for phase in tqdm(DM.phase_list): # 각 phase 별로 모델 학습 및 평가
    if os.path.isdir(f"{dir}/{args.result_name}_{args.testNum}_{phase}") == False:
        os.mkdir(f"{dir}/{args.result_name}_{args.testNum}_{phase}")
    mymodel = MyModel(args.features_n,args.valid_stock_k,args.valid_sector_k,args.each_sector_stock_k,args.final_stock_k,phase,device,args.ensemble,args.clustering)

    mymodel.trainALLSectorModel()

    mymodel.topK_sectors = DM.cluster_list

    mymodel.trainSectorModelsWithValid() # 검증 데이터를 포함하여 모델을 추가 학습

    cagr, sharpe, mdd,_ = mymodel.backtest(verbose=True,agg=args.aggregate,use_all=args.use_all, isTest=False,dir=dir) # 백테스팅 수행
    result[phase] = {"CAGR": cagr, "Sharpe Ratio": sharpe, "MDD": mdd}

    mymodel.save_models(f"{dir}/{args.result_name}_{args.testNum}_{phase}")

pd.DataFrame(result).to_csv(f"{dir}/train_result_file.csv") # 결과 저장