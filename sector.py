import pandas as pd
import os

# 경로 설정
base_path = './data_kr_sector'
symbol_path = os.path.join(base_path, 'symbol.csv')
sector_folder = os.path.join(base_path, '구성종목')

# symbol.csv 불러오기
symbol_df = pd.read_csv(symbol_path)
symbol_df['sector'] = None  # 섹터 컬럼 초기화

# 구성종목 폴더 안의 모든 csv 파일을 확인
for filename in os.listdir(sector_folder):
    if filename.endswith('.csv'):
        sector_name = filename.replace('.csv', '')
        sector_path = os.path.join(sector_folder, filename)

        try:
            sector_df = pd.read_csv(sector_path)
            matching = symbol_df['code'].isin(sector_df['종목코드'])
            symbol_df.loc[matching, 'sector'] = sector_name
        except Exception as e:
            print(f"{sector_name} 처리 중 오류 발생: {e}")

# 결과 저장 (선택사항)
symbol_df.to_csv(os.path.join(base_path, 'symbol_with_sector.csv'), index=False)
print("섹터 정보가 추가된 symbol_with_sector.csv 생성 완료")
