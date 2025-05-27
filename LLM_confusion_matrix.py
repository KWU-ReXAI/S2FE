import csv
import pandas as pd

# ① Confusion Matrix 열 추가 및 필터링
def process_and_label(file_path, output_path):
    result_rows = []

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames + ['Confusion Matrix']

        for row in reader:
            prediction = row.get('LLM 예측', '').strip().lower()
            label = row.get('판단', '').strip().lower()

            # 결측/제외 제거
            if prediction == '' or label == '':
                continue
            if prediction == 'irrelevant' or label == 'excluded':
                continue

            # Confusion Matrix 조건 부여
            if prediction == 'up' and label == 'true':
                row['Confusion Matrix'] = 'PP'
            elif prediction == 'up' and label == 'false':
                row['Confusion Matrix'] = 'PN'
            elif prediction == 'down' and label == 'true':
                row['Confusion Matrix'] = 'NP'
            elif prediction == 'down' and label == 'false':
                row['Confusion Matrix'] = 'NN'
            else:
                continue

            result_rows.append(row)

    # 저장
    with open(output_path, 'w', encoding='utf-8 sig', newline='') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)


def summarize_counts_and_ratios(file_path, output_path):
    df = pd.read_csv(file_path)

    # 1) 개수 집계
    counts = df.groupby(['종목명', 'Confusion Matrix']).size().unstack(fill_value=0)
    for label in ['PP', 'PN', 'NP', 'NN']:
        if label not in counts.columns:
            counts[label] = 0
    counts = counts[['PP', 'PN', 'NP', 'NN']]
    counts['Total'] = counts.sum(axis=1)

    # 2) 비율 계산 (기존)
    ratio_df = counts[['PP', 'PN', 'NP', 'NN']].div(counts['Total'], axis=0)
    ratio_df.columns = ['PP의 비율', 'PN의 비율', 'NP의 비율', 'NN의 비율']

    # 3) 민감도, 특이도 계산
    #    PP + NP 가 0인 경우 NaN 방지
    counts['민감도'] = counts.apply(
        lambda row: row['PP'] / (row['PP'] + row['NP'])
        if (row['PP'] + row['NP']) > 0 else 0, axis=1
    )
    counts['특이도'] = counts.apply(
        lambda row: row['NN'] / (row['NN'] + row['PN'])
        if (row['NN'] + row['PN']) > 0 else 0, axis=1
    )

    # 4) 정답률(PP + NN) 계산
    ratio_df['정답률(PP+NN)'] = ratio_df['PP의 비율'] + ratio_df['NN의 비율']

    # 5) 최종 결합
    final_df = pd.concat([counts, ratio_df], axis=1).reset_index()
    final_df = final_df[
        ['종목명', 'PP', 'PN', 'NP', 'NN', 'Total',
         'PP의 비율', 'PN의 비율', 'NP의 비율', 'NN의 비율',
         '민감도', '특이도', '정답률(PP+NN)']
    ]
    float_cols = ['PP의 비율', 'PN의 비율', 'NP의 비율', 'NN의 비율',
                  '민감도', '특이도', '정답률(PP+NN)']
    final_df[float_cols] = final_df[float_cols].round(2)
    # 정렬 (원하는 기준으로 변경 가능)
    final_df = final_df.sort_values(by="정답률(PP+NN)", ascending=False)

    # 저장
    final_df.to_csv(output_path, index=False, encoding='utf-8 sig')
    print(f"요약 저장 완료: {output_path}")

def get_only_up_code(strdate):
    filepath = f"./preprocessed_data/llm/date_regression/정보기술/{strdate}.csv"
    df = pd.read_csv(filepath)
    # 'predict' 열이 'up'인 경우만 필터링
    df_up = df[df["prediction"] == "up"]
    filepath2 = f"./preprocessed_data/llm/date_regression/산업재/{strdate}.csv"
    df2 = pd.read_csv(filepath)
    # 'predict' 열이 'up'인 경우만 필터링
    df_up2 = df[df["prediction"] == "up"]

    df_up = pd.concat([df_up,df_up2],axis=0)
    return df_up


import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# — 한글 폰트 설정 (필요 시) —
mpl.rc('font', family='Malgun Gothic')
mpl.rcParams['axes.unicode_minus'] = False


def plot_accuracy_distribution(summary_csv_path: str,
                               output_png_path: str,
                               column_name: str = '정답률(PP+NN)', title:str="p"):
    """
    summary_csv_path: 분석할 summary CSV 파일 경로
    output_png_path: 저장할 그래프 파일 경로 (*.png)
    column_name:    히스토그램을 그릴 칼럼명 (0~100 사이 퍼센트 값)
    """
    # 1) CSV 읽기
    df = pd.read_csv(summary_csv_path, encoding='utf-8-sig')

    # 2) 0~1 범위로 변환 (퍼센트 → 비율)
    if df[column_name].max() > 1.0:
        df['accuracy_ratio'] = df[column_name] / 100.0
    else:
        df['accuracy_ratio'] = df[column_name].copy()

    # 3) 히스토그램 그리기
    plt.figure(figsize=(8, 4))
    n, bins, patches = plt.hist(
        df['accuracy_ratio'].dropna(),
        bins=20,
        range=(0.0, 1.0),
        edgecolor='black'
    )
    plt.title(f'정답률 분포 of {title}')
    plt.xlabel('정답률 (비율)')
    plt.ylabel('종목 수')
    plt.xlim(0.0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # 4) 파일로 저장
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    plt.savefig(output_png_path, dpi=300)
    plt.show()
    print(f"히스토그램 저장 완료: {output_png_path}")


# — 사용 예시 —
if __name__ == '__main__':
    summary_csv = './result/result_analysis/Total_confusion_summary_2022_Q4_to_2023_Q3.csv'
    output_png = './result/result_analysis/accuracy_distribution_2022_Q4_to_2023_Q3.png'
    plot_accuracy_distribution(summary_csv, output_png,title="p3")
    summary_csv = './result/result_analysis/Total_confusion_summary_2020_Q4_to_2021_Q3.csv'
    output_png = './result/result_analysis/accuracy_distribution_2020_Q4_to_2021_Q3.png'
    plot_accuracy_distribution(summary_csv, output_png,title="p1")
    process_and_label("./preprocessed_data/llm_cmp_result/LLM_result_total_2022_Q4_to_2023_Q3.csv",
                      "./preprocessed_data/llm_cmp_result/Total_confusion_2022_Q4_to_2023_Q3.csv")
    summarize_counts_and_ratios("./preprocessed_data/llm_cmp_result/Total_confusion_2022_Q4_to_2023_Q3.csv",
                                 "./result/result_analysis/Total_confusion_summary_2022_Q4_to_2023_Q3.csv")

    process_and_label("./preprocessed_data/llm_cmp_result/LLM_result_total_2020_Q4_to_2021_Q3.csv",
                      "./preprocessed_data/llm_cmp_result/Total_confusion_2020_Q4_to_2021_Q3.csv")
    summarize_counts_and_ratios("./preprocessed_data/llm_cmp_result/Total_confusion_2020_Q4_to_2021_Q3.csv",
                                 "./result/result_analysis/Total_confusion_summary_2020_Q4_to_2021_Q3.csv")

"""
if __name__ == "__main__":
    df1=get_only_up_code("2020_Q4")
    df2=get_only_up_code("2021_Q1")
    df3=get_only_up_code("2021_Q2")
    df4=get_only_up_code("2021_Q3")
    df_total = pd.concat([df1,df2,df3,df4],axis=0)
    df_total.to_csv("./preprocessed_data/llm_cmp_result/total_2020_Q4_to_2021_Q3_up.csv",index=False,encoding='utf-8 sig')

    process_and_label("./preprocessed_data/llm_cmp_result/LLM_result_total_2022_Q4_to_2023_Q3.csv", "./preprocessed_data/llm_cmp_result/Total_confusion_2022_Q4_to_2023_Q3.csv")
    summarize_counts_and_ratios2("./preprocessed_data/llm_cmp_result/Total_confusion_2022_Q4_to_2023_Q3.csv", "./result/result_analysis/Total_confusion_summary_2022_Q4_to_2023_Q3.csv")

    process_and_label("./preprocessed_data/llm_cmp_result/LLM_result_total_2020_Q4_to_2021_Q3.csv",
                      "./preprocessed_data/llm_cmp_result/Total_confusion_2020_Q4_to_2021_Q3.csv")
    summarize_counts_and_ratios2("./preprocessed_data/llm_cmp_result/Total_confusion_2020_Q4_to_2021_Q3.csv",
                                 "./result/result_analysis/Total_confusion_summary_2020_Q4_to_2021_Q3.csv")"""