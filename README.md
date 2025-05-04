[오디오 파일]
https://drive.google.com/drive/folders/1y2yT0Tq5XbcGzJMK-QSnUAuOkXdwhJHE?usp=drive_link

[실행 순서]

1. python datapreprocessing.py --isall True

2. python datapreprocessing.py --isall cluster

3. python clustering.py

4. python datapreprocessing.py --isall False

5. python train.py --testNum 5

6. python test.py --testNum 5


[집중적으로 분석해야 하는 파일]

- datapreprocessing.py
- model.py
- datamanager.py
- train.py
- test.py

[실험 결과 보내는 곳]

한 압축폴더에 ./preprocessed_data/ 폴더 전체 & ./result/test_result_dir/ 폴더 전체

jhzzang0703@naver.com



