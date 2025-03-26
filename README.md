# S3CE

[실행순서]

python datapreprocessing2.py --isall cluster

python cluster.py (해당 결과를 datapreprocessing2.py의 if args.isall == "False": 아래의 cluster_list에 복붙해서 클러스터 결과 최신화)

python datapreprocessing2.py --isall True

python datapreprocessing2.py --isall False

python train.py --testNum 5

python test.py --testNum 5