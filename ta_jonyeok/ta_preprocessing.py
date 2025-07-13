import pandas as pd
import talib
import os
import pandas as pd
import os

# 파일이 있는 디렉터리 경로
directory = './data_kr/code_by_sector/'

# 읽어올 파일 이름 리스트
files_to_read = ['산업재.csv', '정보기술.csv']

# 모든 종목 코드를 저장할 리스트
all_codes = []

# 각 파일을 순회하며 종목 코드 추출
for file_name in files_to_read:
    # 파일 경로 조합
    file_path = os.path.join(directory, file_name)

    try:
        # CSV 파일을 DataFrame으로 읽기
        # 'cp949' 인코딩은 한글 파일 깨짐을 방지합니다.
        df = pd.read_csv(file_path, encoding='utf-8-sig')

        # '종목코드' 또는 '코드 번호' 열이 있는지 확인하고 리스트에 추가
        if '종목코드' in df.columns:
            # 종목코드를 6자리 문자열 형식으로 변환하여 리스트에 추가
            codes = df['종목코드'].astype(str).str.zfill(6).tolist()
            all_codes.extend(codes)
            print(f"'{file_name}' 파일에서 {len(codes)}개의 종목 코드를 가져왔습니다.")
        elif '코드 번호' in df.columns:
            # 코드 번호를 6자리 문자열 형식으로 변환하여 리스트에 추가
            codes = df['코드 번호'].astype(str).str.zfill(6).tolist()
            all_codes.extend(codes)
            print(f"'{file_name}' 파일에서 {len(codes)}개의 종목 코드를 가져왔습니다.")
        else:
            print(f"경고: '{file_name}' 파일에 '종목코드' 또는 '코드 번호' 열이 없습니다.")

    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    except Exception as e:
        print(f"'{file_name}' 파일을 처리하는 중 오류가 발생했습니다: {e}")


for code in all_codes:

    # 1. 데이터 불러오기
    try:
        # 사용자님의 실제 파일 경로에 맞게 수정해주세요.
        price_data = pd.read_csv(f"./data_kr/price/{code}.csv")
    except FileNotFoundError:
        print(f"오류: '{code}.csv' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
        exit()

    # 2. 날짜 필터링
    # '날짜' 열을 datetime 형식으로 변환합니다.
    price_data['날짜'] = pd.to_datetime(price_data['날짜'])
    # 2020년 3월 1일 이후의 데이터만 선택합니다.
    filtered_data = price_data[price_data['날짜'] >= '2020-03-01'].copy()

    # 3. 기술적 지표 계산
    # 'talib' 계산에 필요한 데이터를 float 형태로 변환합니다.
    high = filtered_data['고가'].astype(float)
    low = filtered_data['저가'].astype(float)
    close = filtered_data['종가'].astype(float)
    volume = filtered_data['거래량'].astype(float)

    print("기술적 지표를 계산 중입니다...")

    # 다양한 기술적 지표를 계산하여 DataFrame에 새로운 열로 추가합니다.
    filtered_data['SMA_20'] = talib.SMA(close, timeperiod=20)
    filtered_data['SMA_60'] = talib.SMA(close, timeperiod=60)
    filtered_data['EMA_20'] = talib.EMA(close, timeperiod=20)
    filtered_data['RSI'] = talib.RSI(close, timeperiod=14)

    upper, middle, lower = talib.BBANDS(close, timeperiod=20)
    filtered_data['BB_UPPER'] = upper
    filtered_data['BB_MIDDLE'] = middle
    filtered_data['BB_LOWER'] = lower

    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    filtered_data['MACD'] = macd
    filtered_data['MACD_SIGNAL'] = macdsignal
    filtered_data['MACD_HIST'] = macdhist

    slowk, slowd = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3)
    filtered_data['STOCH_SLOWK'] = slowk
    filtered_data['STOCH_SLOWD'] = slowd

    filtered_data['OBV'] = talib.OBV(close, volume)

    print("계산이 완료되었습니다.")

    # 4. 파일 저장
    # 저장할 디렉터리 경로를 설정합니다.
    output_dir = './ta_jonyeok/ta_preprocessed_data'

    # 디렉터리가 존재하지 않으면 새로 생성합니다.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"'{output_dir}' 디렉터리를 생성했습니다.")

    # 처리된 데이터를 CSV 파일로 저장합니다.
    output_path = os.path.join(output_dir, f'{code}.csv')
    filtered_data.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n✅ 처리가 완료되었습니다!")
    print(f"결과 파일이 다음 경로에 저장되었습니다: {output_path}")