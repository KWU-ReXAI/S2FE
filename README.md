[실행 순서]

사용해야 할 데이터 폴더명을 data_kr로 바꿔주세요!

1. python datapreprocessing_total.py --isall True

2. python datapreprocessing_total.py --isall cluster

3. python clustering.py

4. python datapreprocessing_total.py --isall False

5. python train.py --testNum 5

6. python test.py --testNum 5


[집중적으로 분석해야 하는 파일]

- datapreprocessing_total.py
- model.py
- datamanager.py
- train.py
- test.py