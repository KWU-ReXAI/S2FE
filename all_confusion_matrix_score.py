import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import matplotlib
matplotlib.rc('font', family='Malgun Gothic')


# 저장 경로 설정
save_path_root = r"C:\DL\S3CE\preprocessed_data\llm\confusion_matrix\all_confusion_matrix_score"
os.makedirs(save_path_root, exist_ok=True)

# 원본 데이터 위치
root_path = r"C:\DL\S3CE\preprocessed_data\llm\confusion_matrix\confusion_matrix_score"
sectors = ['산업재', '정보기술']
labels = ['PP', 'PN', 'NP', 'NN']

# 카운터 초기화
sector_counters = {sector: Counter() for sector in sectors}
total_counter = Counter()

# 섹터별 confusion_label 수집
for sector in sectors:
    sector_path = os.path.join(root_path, sector)
    if not os.path.isdir(sector_path):
        continue

    for file in os.listdir(sector_path):
        if not file.endswith("_confusion_matrix_score.csv"):
            continue

        file_path = os.path.join(sector_path, file)
        try:
            df = pd.read_csv(file_path)
            labels_list = df['confusion_label'].dropna().tolist()
            sector_counters[sector].update(labels_list)
            total_counter.update(labels_list)
        except Exception as e:
            print(f"[ERROR] {file_path} 읽기 실패: {e}")

# 지표 계산 및 저장 함수
def analyze_and_save(counter: Counter, name: str):
    total = sum(counter.values())
    PP = counter.get('PP', 0)
    PN = counter.get('PN', 0)
    NP = counter.get('NP', 0)
    NN = counter.get('NN', 0)

    # 정확도, 정밀도, 재현율 계산
    accuracy = (PP + NN) / total if total else 0
    precision = PP / (PP + NP) if (PP + NP) else 0
    recall = PP / (PP + PN) if (PP + PN) else 0

    # 데이터프레임 생성
    df = pd.DataFrame({
        'label': labels,
        'count': [counter.get(lbl, 0) for lbl in labels],
        'rate(%)': [round((counter.get(lbl, 0) / total) * 100, 2) if total else 0 for lbl in labels]
    })

    df.loc[len(df)] = ['TOTAL', total, '']
    df.loc[len(df)] = ['정확도(Accuracy)', round(accuracy * 100, 2), '']
    df.loc[len(df)] = ['정밀도(Precision)', round(precision * 100, 2), '']
    df.loc[len(df)] = ['재현율(Recall)', round(recall * 100, 2), '']

    # CSV 저장
    csv_path = os.path.join(save_path_root, f"confusion_matrix_summary_{name}.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"[CSV DONE] {name} → {csv_path}")

    # 시각화
    heat_df = pd.DataFrame([[PP, PN], [NP, NN]], index=['P (예측+)', 'N (예측-)'], columns=['P (실제+)', 'N (실제-)'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(heat_df, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    img_path = os.path.join(save_path_root, f"confusion_matrix_summary_{name}.png")
    plt.savefig(img_path)
    plt.close()
    print(f"[IMG DONE] {name} → {img_path}")

# 저장 실행
for sector in sectors:
    analyze_and_save(sector_counters[sector], sector)

analyze_and_save(total_counter, "total")
