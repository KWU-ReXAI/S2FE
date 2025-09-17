from datamanager import DataManager # datamanager.py 파일에서 DataManager 클래스를 가져옴

dm = DataManager(1,1)

dm.create_date_list()

# for문의 phase 변수(0, 1, 2, 3)를 "p1", "p2", "p3", "p4" 키로 변환
for phase in range(0,4):
    # phase_key를 f-string을 이용해 생성 (e.g., phase가 0일 때 "p1" 생성)
    phase_key = f"p{phase + 1}"

    # 생성한 phase_key로 딕셔너리에 접근
    train_start = dm.phase_list[phase_key][0]
    valid_start = dm.phase_list[phase_key][1]
    test_start = dm.phase_list[phase_key][2]
    test_end = dm.phase_list[phase_key][3]

    print(
        f"train: {dm.pno2date(train_start)} ~ {dm.pno2date(valid_start - 1)} / valid: {dm.pno2date(valid_start)} ~ {dm.pno2date(test_start - 1)}"
        f" / test: {dm.pno2date(test_start)} ~ {dm.pno2date(test_end - 1)}")
    