import OpenDartReader
import pandas as pd
import os
import datetime

s_date = "2015-10-01"
#e_date = "2024-12-31"
e_date = "2025-06-30" # Q1: 1 2 3 / Q2: 4 5 6
# 현재 스크립트의 경로를 기준으로 작업 디렉토리 변경
os.chdir(os.path.dirname(os.path.abspath(__file__)))

api_key = 'a4ccf72e53bf597911d0ff504d58c5f09f2029a3'
#api_key = '84b324165b031bd4d24fc337519ac7ebf126e87e'
dart = OpenDartReader(api_key)

# CSV 파일 읽기 (code 컬럼을 문자열로 읽기)
df_kospi = pd.read_csv("./data_kr/symbol.csv", dtype={'code': str})
codes = df_kospi['code']

# 저장할 디렉토리 지정 및 없으면 생성
output_dir = "./data_kr/public_info"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

start_year = 2015
end_year = 2025

# date_str 형식이 "YYYY-MM-DD"일 때, 분기 문자열을 반환하는 함수
def date2quarter(date_str):
    """예: '2023-07-15' → '2023_Q3'"""
    year, month, _ = map(int, date_str.split("-"))  # "2023-07-15" → (2023, 7, 15)
    if 1 <= month <= 3:
        quarter = "Q1"
    elif 4 <= month <= 6:
        quarter = "Q2"
    elif 7 <= month <= 9:
        quarter = "Q3"
    else:
        quarter = "Q4"
    return f"{quarter}"


def merge_year_quarter_from_csv(csv_path, drop_cols=None, total_option=False):
    if drop_cols is None:
        drop_cols = []

    # CSV 파일 읽기
    df = pd.read_csv(csv_path)

    df = df.fillna(0)

    # total_option이 True이고 "관계" 컬럼이 존재한다면, "관계" 값이 "계"인 행 제거
    if total_option and "이름" in df.columns:
        df = df[df["이름"] != "계"]

    # drop_cols에 지정된 컬럼 제거 (존재하지 않는 컬럼은 무시)
    df = df.drop(columns=drop_cols, errors='ignore')

    # '연도'와 '분기' 컬럼의 타입을 적절하게 변환
    df['연도'] = df['연도'].astype(int)
    df['분기'] = df['분기'].astype(str)

    # '연도'와 '분기'를 제외한 나머지 컬럼들을 숫자형으로 변환 (쉼표 제거 후)
    for col in df.columns:
        if col not in ['연도', '분기']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

    # 그룹화에 사용할 각 열의 평균을 계산할 딕셔너리 생성 ('연도', '분기' 제외)
    agg_dict = {col: 'mean' for col in df.columns if col not in ['연도', '분기']}

    # '연도'와 '분기'별로 그룹화하여 평균 계산
    grouped = df.groupby(['연도', '분기'], as_index=False).agg(agg_dict)

    """# 2015_Q4부터 2024_Q3까지 모든 연도-분기 조합 생성
    all_pairs = []
    for year in range(2015, 2026):  # 2015 ~ 2024년 반복
        if year == 2015:
            quarters = ['Q4']  # 2015년은 Q4만 포함
        elif year == 2025:
            quarters = ['Q1']  # 2024년은 Q1~Q3만 포함
        else:
            quarters = ['Q1', 'Q2', 'Q3', 'Q4']
        for q in quarters:
            all_pairs.append((year, q))"""
    today = datetime.date.today()
    current_year = today.year
    current_quarter = (today.month - 1) // 3 + 1

    # 2015_Q4부터 현재 분기까지 모든 연도-분기 조합을 동적으로 생성
    all_pairs = []
    for year in range(2015, current_year + 1):  # 2015년부터 현재 연도까지 반복
        start_q = 4 if year == 2015 else 1
        end_q = current_quarter if year == current_year else 4

        for q_num in range(start_q, end_q + 1):
            all_pairs.append((year, f"Q{q_num}"))

    # 모든 연도-분기 조합 DataFrame 생성
    full_index_df = pd.DataFrame(all_pairs, columns=['연도', '분기'])

    # 전체 조합과 그룹화 결과를 병합 (왼쪽 조인)
    merged = pd.merge(full_index_df, grouped, on=['연도', '분기'], how='left')

    # 결측값(NaN)을 0으로 채움
    merged = merged.fillna(0)
    merged.drop(columns=['연도', '분기'], errors='ignore', inplace=True)

    return merged



def jeung_ja():
    # 컬럼명을 한국어로 변경하기 위한 사전
    rename_dict = {
        'rcept_no': '접수번호',
        'corp_cls': '법인구분',
        'corp_code': '고유번호',
        'corp_name': '회사명',
        'isu_dcrs_de': '증자일자',        # YYYYMMDD 형태일 수 있음
        'isu_dcrs_stle': '증자방식',
        'isu_dcrs_stock_knd': '증자주식종류',
        'isu_dcrs_qy': '증자주식수',
        'isu_dcrs_mstvdv_fval_amount': '주식액면가액',
        'isu_dcrs_mstvdv_amount': '발행금액',
        'stlm_dt': '결제일시'            # YYYYMMDD 형태일 수 있음
        # 필요한 경우 추가 컬럼을 이곳에 계속 등록
    }

    for company_code in codes:
        company_code = str(company_code).zfill(6)
        all_years_data = []  # 각 회사의 데이터를 저장할 리스트

        for year in range(start_year, end_year + 1):
            try:
                # 해당 연도의 '증자' 보고서 조회
                data = dart.report(company_code, '증자', year)

                # 데이터가 None이거나 DataFrame이 아닐 수 있으므로 처리
                if data is None:
                    continue

                if not isinstance(data, pd.DataFrame):
                    df = pd.DataFrame(data)
                else:
                    df = data

                # 연도 정보를 추가
                df['연도'] = year

                # 리스트에 추가
                all_years_data.append(df)

            except Exception as e:
                print(f"{year}년 데이터 조회 중 오류 발생: {e}")

        # 조회 결과가 있을 경우
        if all_years_data:
            final_df = pd.concat(all_years_data, ignore_index=True)

            # 컬럼명을 한국어로 변경
            final_df.rename(columns=rename_dict, inplace=True)

            # (2) '납입일자'가 존재한다면 → 'YYYYMMDD' → 'YYYY-MM-DD' 변환 후 분기 계산
            if '결제일시' in final_df.columns:
                final_df['결제일시'] = final_df['결제일시'].astype(str).apply(
                    lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:] if len(x) == 8 else x
                )
                final_df['분기'] = final_df['결제일시'].apply(
                    lambda x: date2quarter(x) if len(str(x).split('-')) == 3 else None
                )

            # 저장할 서브 디렉토리 생성 (없으면 생성)
            output_subdir = "./data_kr/public_info/증자"
            os.makedirs(output_subdir, exist_ok=True)

            output_file = os.path.join(output_subdir, f"{company_code}.csv")
            final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"모든 연도의 데이터를 합쳐 {output_file}에 저장했습니다.")
        else:
            print("조회 결과가 없습니다.")

def maximun_juju():
    # 컬럼명을 한국어로 변경하기 위한 사전
    rename_dict = {
        'rcept_no': '접수번호',
        'corp_cls': '법인구분',
        'corp_code': '회사코드',
        'corp_name': '회사명',
        'stock_knd': '주식종류',
        'rm': '비고',
        'nm':'이름',
        'relate':'관계',
        'bsis_posesn_stock_co': '기초 소유 주식 수',
        'bsis_posesn_stock_qota_rt': '기초 소유주식 지분율',
        'trmend_posesn_stock_co': '기말 소유주식 수',
        'trmend_posesn_stock_qota_rt': '기말 소유주식 지분율',
        'stlm_dt':'결제일시'
        # 필요한 경우 추가 컬럼을 이곳에 계속 등록
    }

    for company_code in codes:
        company_code = str(company_code).zfill(6)
        all_years_data = []  # 각 회사의 데이터를 저장할 리스트

        for year in range(start_year, end_year + 1):
            try:
                # 해당 연도의 '증자' 보고서 조회
                data = dart.report(company_code, '최대주주', year)

                # 데이터가 None이거나 DataFrame이 아닐 수 있으므로 처리
                if data is None:
                    continue

                if not isinstance(data, pd.DataFrame):
                    df = pd.DataFrame(data)
                else:
                    df = data

                # 연도 정보를 추가
                df['연도'] = year

                # 리스트에 추가
                all_years_data.append(df)

            except Exception as e:
                print(f"{year}년 데이터 조회 중 오류 발생: {e}")

        # 조회 결과가 있을 경우
        if all_years_data:
            final_df = pd.concat(all_years_data, ignore_index=True)

            # 컬럼명을 한국어로 변경
            final_df.rename(columns=rename_dict, inplace=True)

            # (2) '납입일자'가 존재한다면 → 'YYYYMMDD' → 'YYYY-MM-DD' 변환 후 분기 계산
            if '결제일시' in final_df.columns:
                final_df['결제일시'] = final_df['결제일시'].astype(str).apply(
                    lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:] if len(x) == 8 else x
                )
                final_df['분기'] = final_df['결제일시'].apply(
                    lambda x: date2quarter(x) if len(str(x).split('-')) == 3 else None
                )

            # 저장할 서브 디렉토리 생성 (없으면 생성)
            output_subdir = "./data_kr/public_info/최대주주"
            os.makedirs(output_subdir, exist_ok=True)

            output_file = os.path.join(output_subdir, f"{company_code}.csv")
            final_df.to_csv(output_file, index=False, encoding='utf-8-sig')

            merged_df = merge_year_quarter_from_csv(f"./data_kr/public_info/최대주주/{company_code}.csv",['접수번호','회사코드','법인구분','회사명','주식종류','이름','관계','비고','결제일시'],total_option=True)
            print(f"모든 연도의 데이터를 합쳐 {output_file}에 저장했습니다.")
        else:
            print("조회 결과가 없습니다.")

def maximun_juju_change():
    rename_dict = {
        'rcept_no': '접수번호',
        'corp_cls': '법인구분',
        'corp_code': '회사코드',
        'corp_name': '회사명',
        'change_on':'변동일',
        'mxmm_shrholdr_nm':'최대주주명',
        'posesn_stock_co': '보유 주식 수',
        'qota_rt': '보유 지분율',
        'change_cause':'변동원인',
        'rm': '비고',
        'stlm_dt': '결제일시'
        # 필요한 경우 추가 컬럼을 이곳에 계속 등록
    }
    for company_code in codes:
        company_code = str(company_code).zfill(6)
        all_years_data = []  # 각 회사의 데이터를 저장할 리스트

        for year in range(start_year, end_year + 1):
            try:
                # 해당 연도의 '증자' 보고서 조회
                data = dart.report(company_code, '최대주주변동', year)

                # 데이터가 None이거나 DataFrame이 아닐 수 있으므로 처리
                if data is None:
                    continue

                if not isinstance(data, pd.DataFrame):
                    df = pd.DataFrame(data)
                else:
                    df = data

                # 연도 정보를 추가
                df['연도'] = year

                # 리스트에 추가
                all_years_data.append(df)

            except Exception as e:
                print(f"{year}년 데이터 조회 중 오류 발생: {e}")

        # 조회 결과가 있을 경우
        if all_years_data:
            final_df = pd.concat(all_years_data, ignore_index=True)

            # 컬럼명을 한국어로 변경
            final_df.rename(columns=rename_dict, inplace=True)

            # (2) '납입일자'가 존재한다면 → 'YYYYMMDD' → 'YYYY-MM-DD' 변환 후 분기 계산
            if '결제일시' in final_df.columns:
                final_df['결제일시'] = final_df['결제일시'].astype(str).apply(
                    lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:] if len(x) == 8 else x
                )
                final_df['분기'] = final_df['결제일시'].apply(
                    lambda x: date2quarter(x) if len(str(x).split('-')) == 3 else None
                )

            # 저장할 서브 디렉토리 생성 (없으면 생성)
            output_subdir = "./data_kr/public_info/최대주주변동"
            os.makedirs(output_subdir, exist_ok=True)

            output_file = os.path.join(output_subdir, f"{company_code}.csv")
            final_df.to_csv(output_file, index=False, encoding='utf-8-sig')

            print(f"모든 연도의 데이터를 합쳐 {output_file}에 저장했습니다.")
        else:
            print("조회 결과가 없습니다.")

def employee():
    rename_dict = {
        'rcept_no': '접수번호',
        'corp_cls': '법인구분',
        'corp_code': '회사코드',
        'corp_name': '회사명',
        'sm':'직원 수',
        'rm': '비고',
        'stlm_dt': '결제일시',
        'fyer_salary_totamt':'총 급여액',
        'jan_salary_am':'1인평균급여'
        # 필요한 경우 추가 컬럼을 이곳에 계속 등록
    }
    to_drop = ['sexdstn','fo_bbm','reform_bfe_emp_co_rgllbr','reform_bfe_emp_co_cnttk','reform_bfe_emp_co_etc','rgllbr_co',
               'cnttk_co','rgllbr_abacpt_labrr_co','cnttk_abacpt_labrr_co','avrg_cnwk_sdytrn',]
    for company_code in codes:
        company_code = str(company_code).zfill(6)
        all_years_data = []  # 각 회사의 데이터를 저장할 리스트

        for year in range(start_year, end_year + 1):
            try:
                # 해당 연도의 '증자' 보고서 조회
                data = dart.report(company_code, '직원', year)
                data.drop(to_drop, axis=1, inplace=True)

                # 데이터가 None이거나 DataFrame이 아닐 수 있으므로 처리
                if data is None:
                    continue

                if not isinstance(data, pd.DataFrame):
                    df = pd.DataFrame(data)
                else:
                    df = data

                # 연도 정보를 추가
                df['연도'] = year

                # 리스트에 추가
                all_years_data.append(df)

            except Exception as e:
                print(f"{year}년 데이터 조회 중 오류 발생: {e}")

        # 조회 결과가 있을 경우
        if all_years_data:
            final_df = pd.concat(all_years_data, ignore_index=True)

            # 컬럼명을 한국어로 변경
            final_df.rename(columns=rename_dict, inplace=True)

            # (2) '납입일자'가 존재한다면 → 'YYYYMMDD' → 'YYYY-MM-DD' 변환 후 분기 계산
            if '결제일시' in final_df.columns:
                final_df['결제일시'] = final_df['결제일시'].astype(str).apply(
                    lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:] if len(x) == 8 else x
                )
                final_df['분기'] = final_df['결제일시'].apply(
                    lambda x: date2quarter(x) if len(str(x).split('-')) == 3 else None
                )

            # 저장할 서브 디렉토리 생성 (없으면 생성)
            output_subdir = "./data_kr/public_info/직원현황"
            os.makedirs(output_subdir, exist_ok=True)

            output_file = os.path.join(output_subdir, f"{company_code}.csv")
            final_df.to_csv(output_file, index=False, encoding='utf-8-sig')

            print(f"모든 연도의 데이터를 합쳐 {output_file}에 저장했습니다.")
        else:
            print("조회 결과가 없습니다.")


def prime_juju():
    rename_dict = {
        'rcept_no': '접수번호',
        'rcept_dt': '접수일자',
        'corp_cls': '법인구분',
        'corp_code': '회사코드',
        'corp_name': '회사명',
        'repror': '대표보고자',
        'isu_exctv_rgist_at': '발행 회사 관계 임원(등기여부)',
        'isu_exctv_ofcps': '발행 회사 관계 임원 직위',
        'isu_main_shrholdr': '발행 회사 관계 주요 주주',
        'sp_stock_lmp_cnt': '특정 증권 등 소유 수',
        'sp_stock_lmp_irds_cnt': '특정 증권 등 소유 증감 수',
        'sp_stock_lmp_rate': '특정 증권 등 소유 비율',
        'sp_stock_lmp_irds_rate': '특정 증권 등 소유 증감 비율'
    }

    # 전역 변수 codes가 정의되어 있다고 가정합니다.
    for company_code in codes:
        company_code = str(company_code).zfill(6)

        final_df = dart.major_shareholders_exec(company_code)

        # 데이터가 없거나 None인 경우, 기본 데이터 프레임 생성 (나머지는 0으로 채움)
        if final_df is None or final_df.empty:
            print(f"{company_code}에 대한 데이터가 없습니다. 기본 데이터 형식을 생성합니다.")
            columns_list = [
                '접수번호', '접수일자', '회사코드', '회사명', '대표보고자',
                '발행 회사 관계 임원(등기여부)', '발행 회사 관계 임원 직위',
                '발행 회사 관계 주요 주주', '특정 증권 등 소유 수',
                '특정 증권 등 소유 증감 수', '특정 증권 등 소유 비율',
                '특정 증권 등 소유 증감 비율'
            ]
            # 모든 나머지 컬럼을 0으로 채움
            data_dict = {col: [0] for col in columns_list}
            final_df = pd.DataFrame(data_dict)
            # 기본 데이터에 연도와 분기 추가 (연도: 2015, 분기: Q4)
            final_df['연도'] = 2015
            final_df['분기'] = 'Q4'
        else:
            # 데이터가 있는 경우, 우선 컬럼명 변경 진행
            final_df.rename(columns=rename_dict, inplace=True)

            # 접수일자가 존재하면 날짜 포맷 변환 및 연도/분기 컬럼 추가
            if '접수일자' in final_df.columns:
                final_df['접수일자'] = final_df['접수일자'].astype(str).apply(
                    lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:] if len(x) == 8 else x
                )
                final_df['연도'] = final_df['접수일자'].astype(str).apply(lambda x: x[:4])
                final_df['분기'] = final_df['접수일자'].apply(
                    lambda x: date2quarter(x) if len(str(x).split('-')) == 3 else None
                )
            # 데이터 처리 후 결측값이 있다면 0으로 채움
            final_df.fillna(0, inplace=True)

        # 저장할 서브 디렉토리 생성 (없으면 생성)
        output_subdir = "./data_kr/public_info/주요주주_소유보고"
        os.makedirs(output_subdir, exist_ok=True)

        output_file = os.path.join(output_subdir, f"{company_code}.csv")
        final_df.to_csv(output_file, index=False, encoding='utf-8-sig')

        print(f"데이터를 {output_file}에 저장했습니다.")

def small_juju():
    rename_dict = {
        'rcept_no': '접수번호',
        'corp_cls': '법인구분',
        'corp_code': '회사코드',
        'corp_name': '회사명',
        'se':'구분',
        'shrholdr_co':'주주 수',
        'shrholdr_tot_co': '전체 주주 수',
        'shrholdr_rate':'주주 비율',
        'hold_stock_co':'보유 주식 수',
        'stock_tot_co':'전체 주식 수',
        'hold_stock_rate':'보유 주식 비율',
        'stlm_dt': '결제일시'
        # 필요한 경우 추가 컬럼을 이곳에 계속 등록
    }
    for company_code in codes:
        company_code = str(company_code).zfill(6)
        all_years_data = []  # 각 회사의 데이터를 저장할 리스트

        for year in range(start_year, end_year + 1):
            try:
                # 해당 연도의 '증자' 보고서 조회
                data = dart.report(company_code, '소액주주', year)
                #data.drop(to_drop, axis=1, inplace=True)

                # 데이터가 None이거나 DataFrame이 아닐 수 있으므로 처리
                if data is None:
                    continue

                if not isinstance(data, pd.DataFrame):
                    df = pd.DataFrame(data)
                else:
                    df = data

                # 연도 정보를 추가
                df['연도'] = year

                # 리스트에 추가
                all_years_data.append(df)

            except Exception as e:
                print(f"{year}년 데이터 조회 중 오류 발생: {e}")

        # 조회 결과가 있을 경우
        if all_years_data:
            final_df = pd.concat(all_years_data, ignore_index=True)

            # 컬럼명을 한국어로 변경
            final_df.rename(columns=rename_dict, inplace=True)

            # (2) '납입일자'가 존재한다면 → 'YYYYMMDD' → 'YYYY-MM-DD' 변환 후 분기 계산
            if '결제일시' in final_df.columns:
                final_df['결제일시'] = final_df['결제일시'].astype(str).apply(
                    lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:] if len(x) == 8 else x
                )
                final_df['분기'] = final_df['결제일시'].apply(
                    lambda x: date2quarter(x) if len(str(x).split('-')) == 3 else None
                )

            # 저장할 서브 디렉토리 생성 (없으면 생성)
            output_subdir = "./data_kr/public_info/소액주주"
            os.makedirs(output_subdir, exist_ok=True)

            output_file = os.path.join(output_subdir, f"{company_code}.csv")
            final_df.to_csv(output_file, index=False, encoding='utf-8-sig')

            print(f"모든 연도의 데이터를 합쳐 {output_file}에 저장했습니다.")
        else:
            print(f"{company_code}에 대한 데이터가 없습니다. 기본 데이터 형식을 생성합니다.")
            columns_list = [
                '접수번호','법인구분','회사코드','회사명','구분','주주 ','전체 주주 수','주주 비율','보유 주식 수','전체 주식 수','보유 주식 비율','결제일시','연도','분기'
            ]
            # 모든 나머지 컬럼을 0으로 채움
            data_dict = {col: [0] for col in columns_list}
            final_df = pd.DataFrame(data_dict)
            # 기본 데이터에 연도와 분기 추가 (연도: 2015, 분기: Q4)
            final_df['연도'] = 2015
            final_df['분기'] = 'Q4'

            output_subdir = "./data_kr/public_info/소액주주"
            os.makedirs(output_subdir, exist_ok=True)

            output_file = os.path.join(output_subdir, f"{company_code}.csv")
            final_df.to_csv(output_file, index=False, encoding='utf-8-sig')

def total_jusik():
    rename_dict = {
        'rcept_no': '접수번호',
        'corp_cls': '법인구분',
        'corp_code': '회사코드',
        'corp_name': '회사명',
        'se': '구분',
        'isu_stock_totqy':'발행할 주식 총수',
        'now_to_isu_stock_totqy':'현재까지 발행한 주식 총수',
        'now_to_dcrs_stock_totqy':'현재까지 감소한 주식의 총수',
        'redc':'주식 수 감소',
        'profit_incnr':'이익 소각',
        'rdmstk_repy':'상환주식의 상환',
        'etc':'기타',
        'istc_totqy':'발행주식의 총수',
        'tesstk_co':'자기주식수',
        'distb_stock_co':'유통주식수',
        'stlm_dt': '결제일시'
        # 필요한 경우 추가 컬럼을 이곳에 계속 등록
    }
    for company_code in codes:
        company_code = str(company_code).zfill(6)
        all_years_data = []  # 각 회사의 데이터를 저장할 리스트

        for year in range(start_year, end_year + 1):
            try:
                # 해당 연도의 '증자' 보고서 조회
                data = dart.report(company_code, '주식총수', year)
                # data.drop(to_drop, axis=1, inplace=True)

                # 데이터가 None이거나 DataFrame이 아닐 수 있으므로 처리
                if data is None:
                    continue

                if not isinstance(data, pd.DataFrame):
                    df = pd.DataFrame(data)
                else:
                    df = data

                # 연도 정보를 추가
                df['연도'] = year

                # 리스트에 추가
                all_years_data.append(df)

            except Exception as e:
                print(f"{year}년 데이터 조회 중 오류 발생: {e}")

        # 조회 결과가 있을 경우
        if all_years_data:
            final_df = pd.concat(all_years_data, ignore_index=True)

            # 컬럼명을 한국어로 변경
            final_df.rename(columns=rename_dict, inplace=True)

            # (2) '납입일자'가 존재한다면 → 'YYYYMMDD' → 'YYYY-MM-DD' 변환 후 분기 계산
            if '결제일시' in final_df.columns:
                final_df['결제일시'] = final_df['결제일시'].astype(str).apply(
                    lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:] if len(x) == 8 else x
                )
                final_df['분기'] = final_df['결제일시'].apply(
                    lambda x: date2quarter(x) if len(str(x).split('-')) == 3 else None
                )

            # 저장할 서브 디렉토리 생성 (없으면 생성)
            output_subdir = "./data_kr/public_info/주식총수"
            os.makedirs(output_subdir, exist_ok=True)

            output_file = os.path.join(output_subdir, f"{company_code}.csv")
            final_df.to_csv(output_file, index=False, encoding='utf-8-sig')

            print(f"모든 연도의 데이터를 합쳐 {output_file}에 저장했습니다.")
        else:
            print(f"{company_code}에 대한 데이터가 없습니다. 기본 데이터 형식을 생성합니다.")
            columns_list = [
                '접수번호', '법인구분', '회사코드', '회사명', '구분', '발행할 주식 총수', '현재까지 발행한 주식 총수',
                '현재까지 감소한 주식의 총수', '주식 수 감소', '이익 소각', '상환주식의 상환', '기타', '발행주식의 총수',
                '자기주식수', '유통주식수', '결제일시', '연도'
            ]
            # 모든 나머지 컬럼을 0으로 채움
            data_dict = {col: [0] for col in columns_list}
            final_df = pd.DataFrame(data_dict)
            # 기본 데이터에 연도와 분기 추가 (연도: 2015, 분기: Q4)
            final_df['연도'] = 2015
            final_df['분기'] = 'Q4'

            output_subdir = "./data_kr/public_info/주식총수"
            os.makedirs(output_subdir, exist_ok=True)

            output_file = os.path.join(output_subdir, f"{company_code}.csv")
            final_df.to_csv(output_file, index=False, encoding='utf-8-sig')

def concat_public_info():
    code = "000080"
    df_small = merge_year_quarter_from_csv(f"./data_kr/public_info/소액주주/{code}.csv",['접수번호','법인구분','회사코드','회사명','구분','결제일시'],False)
    df_total = merge_year_quarter_from_csv(f"./data_kr/public_info/주식총수/{code}.csv",['접수번호','법인구분','회사코드','회사명','구분','결제일시'],False)
    df_prime = merge_year_quarter_from_csv(f"./data_kr/public_info/주요주주_소유보고/{code}.csv",['접수번호','법인구분','회사코드','회사명','대표보고자','발행 회사 관계 임원(등기여부)','발행 회사 관계 임원 직위','발행 회사 관계 주요 주주','결제일시'],False)
    df_jeungja = merge_year_quarter_from_csv(f"./data_kr/public_info/증자/{code}.csv",['접수번호','법인구분','회사코드','회사명','증자일자','증자방식','증자주식종류','구분','결제일시'],False)
    df_employee = merge_year_quarter_from_csv(f"./data_kr/public_info/직원현황/{code}.csv", ['접수번호', '법인구분', '회사코드', '회사명', '직원 수','총 급여액','비고', '결제일시'],False)
    df_maximum = merge_year_quarter_from_csv(f"./data_kr/public_info/최대주주/{code}.csv", ['접수번호', '법인구분', '회사코드', '회사명','주식종류','이름','관계', '비고', '결제일시'],True)
    df_maximum_change = merge_year_quarter_from_csv(f"./data_kr/public_info/최대주주변동/{code}.csv", ['접수번호', '법인구분', '회사코드', '회사명','변동일','최대주주명','보유 주식 수','변동원인', '비고', '결제일시'],False)

    dfs = [df_small,df_total,df_prime,df_jeungja,df_employee,df_maximum,df_maximum_change]
    df_concated = pd.concat(dfs,axis=1)

    return df_concated

if __name__ == "__main__":
    jeung_ja()
    maximun_juju()
    maximun_juju_change()
    employee()
    prime_juju()
    small_juju()
    total_jusik()