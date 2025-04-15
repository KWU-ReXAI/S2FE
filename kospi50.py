import OpenDartReader
import pandas as pd
import os
import time

import requests
import pandas as pd

# OpenDART API 키와 조회 대상 정보
api_key = 'a4ccf72e53bf597911d0ff504d58c5f09f2029a3'
corp_code = "000050"
year = 2021
reprt_code = "11011"  # 예: 사업보고서

# 1) 재무제표 데이터 조회 (기존 finstate 호출 가정)
fin_params = {'crtfc_key': api_key, 'corp_code': corp_code,
              'bsns_year': year, 'reprt_code': reprt_code}
res = requests.get('https://opendart.fss.or.kr/api/fnlttSinglAcnt.json', params=fin_params)
df_fin = pd.DataFrame(res.json()['list'])  # finstate 결과 (재무제표 데이터)
# df_fin에는 rcept_no, account_nm, thstrm_amount 등 재무제표 항목들이 포함됨

# 2) 공시 목록 조회하여 제출일 가져오기
list_params = {
    'crtfc_key': api_key,
    'corp_code': corp_code,
    'bgn_de': f"{year}0101",       # 해당 연도의 시작일
    'end_de': f"{year+1}1231",     # 다음년도 말까지 (사업보고서의 경우 다음해 3월말쯤 제출됨)
    'pblntf_ty': 'A',             # 정기공시
    'pblntf_detail_ty': 'A001',   # 사업보고서 (필요에 따라 A002, A003)
    'last_reprt_at': 'N',         # 정정시 최종본만
    'page_count': 100
}
res_list = requests.get('https://opendart.fss.or.kr/api/list.json', params=list_params)
df_list = pd.DataFrame(res_list.json()['list'])

# 3) finstate 결과와 공시일 데이터 병합 (접수번호 rcept_no 기준)
df_merge = pd.merge(df_fin, df_list[['rcept_no', 'rcept_dt']], on='rcept_no', how='left')
# 이제 df_merge의 각 재무제표 항목에 'rcept_dt' 컬럼으로 공시일자(제출일)가 추가됨
