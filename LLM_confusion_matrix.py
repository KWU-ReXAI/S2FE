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
    with open(output_path, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)


# ② 종목별 Confusion Matrix 개수 + 비율 요약 저장 (+ 정답률)
def summarize_counts_and_ratios(file_path, output_path):
    df = pd.read_csv(file_path)

    # 개수 집계
    counts = df.groupby(['종목명', 'Confusion Matrix']).size().unstack(fill_value=0)
    for label in ['PP', 'PN', 'NP', 'NN']:
        if label not in counts.columns:
            counts[label] = 0

    counts = counts[['PP', 'PN', 'NP', 'NN']]
    counts['Total'] = counts.sum(axis=1)

    # 비율 계산
    ratio_df = counts[['PP', 'PN', 'NP', 'NN']].div(counts['Total'], axis=0)
    ratio_df.columns = ['PP의 비율', 'PN의 비율', 'NP의 비율', 'NN의 비율']

    # 정답률(PP + NN) 계산
    ratio_df['정답률(PP+NN)'] = ratio_df['PP의 비율'] + ratio_df['NN의 비율']

    # 최종 결합
    final_df = pd.concat([counts, ratio_df], axis=1).reset_index()
    final_df = final_df[['종목명', 'PP', 'PN', 'NP', 'NN', 'Total',
                         'PP의 비율', 'PN의 비율', 'NP의 비율', 'NN의 비율', '정답률(PP+NN)']]

    final_df["정답률(PP+NN)"] = final_df["정답률(PP+NN)"].astype(float)
    final_df = final_df.sort_values(by="정답률(PP+NN)", ascending=False)

    # 저장
    final_df.to_csv(output_path, index=False, encoding='cp949')

process_and_label("LLM_result_산업재.csv", "산업재_confusion_labeled.csv")
process_and_label("LLM_result_정보기술.csv", "정보기술_confusion_labeled.csv")

summarize_counts_and_ratios("산업재_confusion_labeled.csv", "산업재_confusion_summary.csv")
summarize_counts_and_ratios("정보기술_confusion_labeled.csv", "정보기술_confusion_summary.csv")
