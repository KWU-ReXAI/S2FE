"""import pandas as pd

dir = f"./result/result_S3CE_SectorAll/test_result_dir/test_result_file.csv"  # 훈련 후 검증결과과 저장파일
df = pd.read_csv(dir, header=[0, 1], index_col=0)
df_avg = df["Average"].mean(axis=1)

print(f"CAGR: {df_avg['CAGR']}, SHARPE RATIO: {df_avg['Sharpe Ratio']}, MDD: {df_avg['MDD']}")"""

import pandas as pd
import json


# analyze_csv_differences 함수는 그대로 사용합니다.
def analyze_csv_differences(file_path1: str, file_path2: str, unique_column: str = None) -> dict:
    """
    두 개의 CSV 파일을 읽어 그 차이점을 분석하고 결과를 딕셔너리로 반환합니다.
    (기존 함수와 동일)
    """
    try:
        df1 = pd.read_csv(file_path1)
        df2 = pd.read_csv(file_path2)
    except FileNotFoundError as e:
        return {"error": f"파일을 찾을 수 없습니다: {e}"}

    results = {}

    if df1.shape != df2.shape:
        results['shape_difference'] = {'file1': df1.shape, 'file2': df2.shape}

    if set(df1.columns) != set(df2.columns):
        results['column_difference'] = {
            'only_in_file1': sorted(list(set(df1.columns) - set(df2.columns))),
            'only_in_file2': sorted(list(set(df2.columns) - set(df1.columns)))
        }

    if unique_column and unique_column in df1.columns and unique_column in df2.columns:
        merged = pd.merge(df1, df2, on=[unique_column], how='outer', indicator=True, suffixes=('_file1', '_file2'))
        unique_to_file1 = merged[merged['_merge'] == 'left_only']
        unique_to_file2 = merged[merged['_merge'] == 'right_only']
        if not unique_to_file1.empty or not unique_to_file2.empty:
            results['unique_rows'] = {
                'file1_only_count': len(unique_to_file1),
                'file2_only_count': len(unique_to_file2),
                'file1_only_rows': unique_to_file1[
                    [unique_column] + [c for c in df1.columns if c != unique_column]].reset_index(drop=True),
                'file2_only_rows': unique_to_file2[
                    [unique_column] + [c for c in df2.columns if c != unique_column]].reset_index(drop=True)
            }

        common_ids = merged[merged['_merge'] == 'both'][unique_column]
        df1_common = df1[df1[unique_column].isin(common_ids)].set_index(unique_column).sort_index()
        df2_common = df2[df2[unique_column].isin(common_ids)].set_index(unique_column).sort_index()

        common_cols = sorted(list(set(df1_common.columns) & set(df2_common.columns)))
        diff = df1_common[common_cols].compare(df2_common[common_cols], keep_equal=False, align_axis=1)
        if not diff.empty:
            diff.columns = diff.columns.set_levels(['file1', 'file2'], level=1)
            results['value_differences'] = diff
    else:
        diff = df1.compare(df2, keep_equal=False, align_axis=1)
        if not diff.empty:
            diff.columns = diff.columns.set_levels(['file1', 'file2'], level=1)
            results['value_differences'] = diff

    return results


# [수정된 부분] pretty_print_diff 함수
def pretty_print_diff(results):
    """결과 딕셔너리를 보기 좋게 출력합니다. 결과가 없으면 파일이 동일함을 알립니다."""
    # 결과 딕셔너리가 비어있는지 먼저 확인
    if not results:
        print("✅ 두 파일의 내용이 완전히 동일합니다.")
        return

    for key, value in results.items():
        print(f"--- {key.replace('_', ' ').upper()} ---")
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}:")
                if isinstance(sub_value, pd.DataFrame) and not sub_value.empty:
                    print(sub_value.to_string())
                else:
                    print(f"    {sub_value}")
        elif isinstance(value, pd.DataFrame) and not value.empty:
            print(value.to_string())
        else:
            print(f"  {value}")
        print()


# 함수 호출 및 결과 출력 부분은 그대로 유지
analysis_result = analyze_csv_differences('./analysis/df_data_cluster_0_now.csv', './analysis/df_data_cluster_0.csv',
                                          unique_column='ProductID')
pretty_print_diff(analysis_result)