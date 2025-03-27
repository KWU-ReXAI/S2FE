python datapreprocessing2.py --isall True

python datapreprocessing2.py --isall cluster

python clustering.py

clustering.py 결과 복사해서 datapreprocessing2.py의 --isall False 밑 cluster_list에 추가

python datapreprocessing2.py --isall False

python train.py --testNum 5

python test.py --testNum 5