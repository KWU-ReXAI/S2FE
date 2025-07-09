import argparse
import pandas as pd
import torch
import warnings
import matplotlib.pyplot as plt
from datamanager import DataManager
from model import MyModel
#from model_RF import RF_Model
from tqdm import tqdm
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

cluster_n = 5

warnings.filterwarnings(action='ignore')

parser = argparse.ArgumentParser() # 입력 받을 하이퍼파라미터 설정

parser.add_argument('--features_n',type=int,nargs='?',default=6) # 사용할 재무지표(feature) 개수
parser.add_argument('--valid_sector_k',type=int,nargs='?',default=2) # 검증 단계에서 선택할 상위 섹터 개수
parser.add_argument('--valid_stock_k',type=int,nargs='?',default=5) # 검증 단계에서 선택할 상위 주식 개수
parser.add_argument('--each_sector_stock_k',type=int,nargs='?',default=5) # 각 섹터에서 선택할 주식 개수
parser.add_argument('--final_stock_k',type=int,nargs='?',default=10) # 최종적으로 선택할 주식 개수
parser.add_argument('--result_name',type=str,nargs='?',default="train_result_model") # 결과 파일명
parser.add_argument('--dir_name',type=str,nargs='?',default="train_result_dir") # 결과 디렉토리 명
parser.add_argument('--use_all',type=str,nargs="?",default="SectorAll") # 모델을 평가하는 방식 설정
parser.add_argument('--ensemble',type=str,nargs="?",default="S3CE")
parser.add_argument('--data',type=str,nargs="?",default="All")
parser.add_argument('--clustering',action="store_true",default=True) # 클러스터링 여부
parser.add_argument('--testNum',type=int,nargs='?',default=1) # 클러스터링 여부
parser.add_argument('--Validation',action="store_true") # 클러스터링 여부


parser.add_argument('--lr_MLP',type=float,nargs='?',default=0.01) # 0.01
parser.add_argument('--lr_anfis',type=float,nargs='?',default=0.0001) # 0.0001
parser.add_argument('--epochs_MLP',type=int,nargs='?',default=300) # 300
parser.add_argument('--epochs_anfis',type=int,nargs='?',default=300) # 300
parser.add_argument('--hidden',type=int,nargs='?',default=128) # 128


parser.add_argument('--agg',type=str,nargs='?',default="inter") # inter
parser.add_argument('--inter_n',type=float,nargs='?',default=0.1) # 0.1

args = parser.parse_args()

if isinstance(args.inter_n, float) and args.inter_n.is_integer():
    args.inter_n = int(args.inter_n)

DM = DataManager(features_n= args.features_n, cluster_n=cluster_n)
DM.create_date_list()
result = {}
result_ks = {}
if args.data == "All":
    folder_name = f"result_{args.ensemble}_{args.use_all}"
else:
    folder_name = f"result_{args.ensemble}_{args.use_all}_{args.data}"
dir = f"./result"
if os.path.isdir(dir) == False:
    os.mkdir(dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
file_path = f"./result/{folder_name}/train_parameter.csv"
os.makedirs(os.path.dirname(file_path), exist_ok=True)
df = pd.DataFrame(columns=["Parameter"," ", "Value"])
df.to_csv(file_path, index=False)
recordmodel = MyModel(args.features_n, args.valid_stock_k, args.valid_sector_k, args.each_sector_stock_k,
                          args.final_stock_k, " ", device, args.ensemble, args.clustering, lr_anfis=args.lr_anfis,lr_MLP=args.lr_MLP,epochs_MLP=args.epochs_MLP,epochs_anfis=args.epochs_anfis,hidden=args.hidden)
recordmodel.recordParameter(file_path)

for trainNum in range(0, args.testNum):
    print(f"\nTrain for Train_Model_{trainNum+1}")
    dir = f"./result/{folder_name}/{args.dir_name}_{trainNum+1}"
    if os.path.isdir(dir) == False:
        os.mkdir(dir)
    for phase in tqdm(DM.phase_list):
        print(f"\nTrain Phase of Model {args.ensemble} {trainNum+1}: {phase}")
        if os.path.isdir(f"{dir}/{args.result_name}_{trainNum+1}_{phase}") == False:
            os.mkdir(f"{dir}/{args.result_name}_{trainNum+1}_{phase}")
        if args.ensemble == "S3CE" or args.ensemble == "agg3":
            mymodel = MyModel(args.features_n, args.valid_stock_k, args.valid_sector_k, args.each_sector_stock_k,
                          args.final_stock_k, phase, device, args.ensemble, args.clustering, cluster_n=cluster_n, lr_anfis=args.lr_anfis,lr_MLP=args.lr_MLP,epochs_MLP=args.epochs_MLP,epochs_anfis=args.epochs_anfis,hidden=args.hidden)
        """elif args.ensemble == "RF":
            mymodel = RF_Model(args.features_n, args.valid_stock_k, args.valid_sector_k, args.each_sector_stock_k,
                          args.final_stock_k, phase, args.ensemble,device)"""

        if args.ensemble == "buyhold":
            cagr, sharpe, mdd, _, cagr_ks, sharpe_ks, mdd_ks = mymodel.backtest_BuyHold(verbose=True, withValidation=not args.Validation)

        else:
            if args.use_all == "All":
                mymodel.trainALLSectorModels(withValidation=not args.Validation,model=args.ensemble)
            elif args.use_all == "Sector":
                mymodel.trainClusterModels(withValidation=not args.Validation,model=args.ensemble)
            else:
                print(f"Train Start: cluster_n={cluster_n}, use_all={args.use_all}, lr_anfis={args.lr_anfis}, lr_MLP={args.lr_MLP}, epochs_MLP={args.epochs_MLP}, epochs_anfis={args.epochs_anfis}, hidden={args.hidden}, withValidation={not args.Validation}")
                mymodel.trainALLSectorModels(withValidation=not args.Validation,model=args.ensemble)
                mymodel.trainClusterModels(withValidation=not args.Validation,model=args.ensemble)
            cagr, sharpe, mdd, _, cagr_ks, sharpe_ks, mdd_ks = mymodel.backtest(verbose=True, agg=args.agg,
                                                                                inter_n=args.inter_n,
                                                                                use_all=args.use_all,
                                                                                withValidation=not args.Validation,
                                                                                isTest=False, dir=dir,model=args.ensemble)

        result[phase] = {
            "CAGR": cagr,
            "Sharpe Ratio": sharpe,
            "MDD_model": mdd,
        }
        result_ks[phase] = {"CAGR": cagr_ks,
                            "Sharpe Ratio": sharpe_ks,
                            "MDD": mdd_ks}

        mymodel.save_models(f"{dir}/{args.result_name}_{trainNum+1}_{phase}")
        print(f"Save Model at {dir}/{args.result_name}_{trainNum+1}_{phase}\n")

    result_df = pd.DataFrame(result)
    result_df["Average"] = result_df.mean(axis=1)
    plt.rcParams['axes.unicode_minus'] = False
    plot_columns = [col for col in result_df.columns if col != "Average"]

    plt.figure(figsize=(10, 6))
    for indicator in result_df.index:
        plt.plot(plot_columns, result_df.loc[indicator, plot_columns],
                 marker='o', label=indicator)

    plt.xlabel("Phase")
    plt.ylabel("Evaluation indicator values")
    plt.title("Changes in evaluation indicators by phase")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{dir}/train_result_graph.png", dpi=300, bbox_inches="tight", pad_inches=0.1)

    result_df.to_csv(f"{dir}/train_result_file_{trainNum+1}.csv", encoding='utf-8-sig')

result_df_ks = pd.DataFrame(result_ks)
result_df_ks["Average"] = result_df_ks.mean(axis=1)
plot_columns = [col for col in result_df_ks.columns if col != "Average"]
plt.figure(figsize=(10, 6))
for indicator in result_df_ks.index:
    plt.plot(plot_columns, result_df_ks.loc[indicator, plot_columns],marker='o', label=indicator)

plt.xlabel("Phase")
plt.ylabel("Evaluation indicator values")
plt.title("Changes in evaluation indicators by phase")
plt.legend()
plt.grid(True)
plt.savefig(f"./result/{folder_name}/result_KOSPI_graph.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
result_df_ks.to_csv(f"./result/{folder_name}/result_KOSPI_file.csv", encoding='utf-8-sig')


