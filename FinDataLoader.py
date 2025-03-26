import os
import pandas as pd
os.chdir(os.path.dirname(os.path.abspath(__file__)))
class FinDataLoader:
    def __init__(self,path):
        self.path = path
        self.stock_list = {}
        
        with open(f"{path}/kospi_200.txt", 'r',encoding='utf-8 sig') as f:
            for line in f:
                code, name, sector = line.strip().split(',')
                self.stock_list[code] = name
                
    def get_statement(self, code, year, quarter):
        
        if os.path.isfile(f"{self.path}/{code}/{code}_{year}.csv"):
            df_fs = pd.read_csv(f"{self.path}/{code}/{code}_{year}.csv")
            
            return df_fs[df_fs['Quarter'] == quarter]
        else:
            print(f"파일이 존재하지 않습니다: {self.path}/{code}/{code}_{year}.csv")
            return pd.DataFrame()

    def get_data(self, column="", path=""):
        df_file = pd.read_csv(f"{path}")
        df_data = df_file[column].unique()

        return df_data