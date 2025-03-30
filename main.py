import pandas as pd
import os

class DataManager:
    def __init__(self, features_n):
        self.features_n = features_n
        self.ticker_list = sorted(pd.read_csv("./data_kr/symbol.csv")["code"].tolist())
        self.sector_list = pd.read_csv("./data_kr/symbol.csv")["sector"].tolist()
        self.cluster_list = ["cluster_" + str(i) for i in range(3)]  # 클러스터 번호 부여

        self.phase_list = {"p1": [1, 15, 19, 23], "p2": [5, 19, 23, 27], "p3": [9, 23, 27, 31], "p4": [13, 27, 31, 35]}

    def create_date_list(self):
        # 예시: merged 폴더의 파일 이름이 "2015_Q1.csv", "2015_Q2.csv" 등이라면
        files = os.listdir("./data_kr/date_regression")
        dates = [f.replace(".csv", "") for f in files if f.endswith(".csv")]
        dates = sorted(dates)  # 날짜 순으로 정렬 (적절한 정렬 기준을 적용)
        self.date_list = dates
        return dates

    def get_sector_list(self):
        return self.sector_list
    def get_ticker_list(self):
        return self.ticker_list

    def date2quarter(date_str):
        year, month, _ = map(int, date_str.split("-"))  # "2023-07-15" → (2023, 7, 15)

        # 월에 따라 분기 결정
        if 1 <= month <= 3:
            quarter = "Q1"
        elif 4 <= month <= 6:
            quarter = "Q2"
        elif 7 <= month <= 9:
            quarter = "Q3"
        else:
            quarter = "Q4"

        return f"{year}_{quarter}"

    def quarter2date(quarter_str):
        year, q = quarter_str.split("_")  # "2018_Q3" → ["2018", "Q3"]
        year = int(year)

        # 분기별 시작 & 종료 날짜 설정
        quarter_dict = {
            "Q1": ("01-01", "03-31"),
            "Q2": ("04-01", "06-30"),
            "Q3": ("07-01", "09-30"),
            "Q4": ("10-01", "12-31"),
        }

        if q in quarter_dict:
            start_date = f"{year}-{quarter_dict[q][0]}"  # 시작일
            end_date = f"{year}-{quarter_dict[q][1]}"  # 종료일
            return start_date, end_date
        else:
            raise ValueError("Invalid quarter format. Use YYYY_QX (e.g., 2018_Q3)")

    def pno2date(self, pno: int) -> str:
        try:
            return self.date_list[pno]
        except IndexError:
            raise ValueError(f"pno {pno}은 date_list의 범위를 벗어났습니다. date_list 길이: {len(self.date_list)}")

# phase_list의 모든 pno 값을 pno2date에 적용하여 출력하는 코드
dm = DataManager(features_n=3)
dm.create_date_list()

# phase_list에 포함된 모든 pno 값을 변환
for phase, pno_list in dm.phase_list.items():
    print(f"Phase: {phase}")
    for pno in pno_list:
        try:
            date = dm.pno2date(pno)
            print(f"  pno {pno} -> {date}")
        except ValueError as e:
            print(f"  Error for pno {pno}: {e}")
