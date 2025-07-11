import matplotlib.pyplot as plt

def cagr():
    # Windows 한글폰트 설정 및 기본 폰트 크기/두께 조정
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['font.sans-serif'] = ['Malgun Gothic']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 14  # 전체 기본 폰트 크기
    plt.rcParams['axes.titlesize'] = 18  # 제목 폰트 크기
    plt.rcParams['axes.titleweight'] = 'bold'  # 제목 두께
    plt.rcParams['axes.labelsize'] = 25  # 축 라벨 폰트 크기
    plt.rcParams['axes.labelweight'] = 'bold'  # 축 라벨 두께
    plt.rcParams['xtick.labelsize'] = 25  # x축 눈금 폰트 크기
    plt.rcParams['ytick.labelsize'] = 25  # y축 눈금 폰트 크기

    # 1) 종목 선택 비율 vs CAGR
    selection_ratios = [10, 15, 20, 25, 30]
    cagr_selection = [14.78, -8.14, -8.46, -8.97, -9.22]

    plt.figure(figsize=(8, 6))
    plt.plot(
        selection_ratios, cagr_selection,
        marker='s', linestyle='-', color='blue',
        markerfacecolor='red', markeredgecolor='red',
        markersize=12,  # 마커 크기 키우기
        markeredgewidth=2,  # 마커 테두리 두께
        linewidth=3  # 선 두께
    )
    plt.xlabel('종목 선택 비율 (%)', labelpad=15)
    plt.ylabel('CAGR (%)')
    plt.xticks(selection_ratios)
    plt.ylim(-15, 20)
    plt.yticks(list(range(-15, 21, 5)))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # 2) 특징 선택 개수 vs CAGR
    feature_counts = [3, 4, 5, 6, 7]
    cagr_features = [-13.42, 5.00, 6.11, 16.23, -2.79]

    plt.figure(figsize=(8, 6))
    plt.plot(
        feature_counts, cagr_features,
        marker='s', linestyle='-', color='blue',
        markerfacecolor='red', markeredgecolor='red',
        markersize=12,  # 마커 크기 키우기
        markeredgewidth=2,  # 마커 테두리 두께
        linewidth=3  # 선 두께
    )
    plt.xlabel('특징 선택 개수', labelpad=15)
    plt.ylabel('CAGR (%)')
    plt.xticks(feature_counts)
    plt.ylim(-15, 20)
    plt.yticks(list(range(-15, 21, 5)))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def bar_graph():
    import matplotlib.pyplot as plt

    import matplotlib.pyplot as plt

    # — Windows 한글폰트 설정 및 기본 폰트 크기/두께 조정 —
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['font.sans-serif'] = ['Malgun Gothic']
    plt.rcParams['axes.unicode_minus'] = False

    plt.rcParams['font.size'] = 21  # 전체 기본 폰트 크기
    plt.rcParams['axes.titlesize'] = 18  # 제목 폰트 크기
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 21  # 축 라벨 폰트 크기
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 20  # x축 눈금 폰트 크기
    plt.rcParams['ytick.labelsize'] = 21  # y축 눈금 폰트 크기

    # ──────────────────────────────────────────────────────────────────────────
    # 1) 데이터 조합 바 차트
    labels = [
        'F', 'D', 'M',
        'D+M', '재무제표\n거시경제지표', '재무재표\n공시',
        '재무+공시+거시'
    ]
    cagr_values = [-8.32, -12.07, 5.40, 6.30, 7.77, -11.99, 10.18]

    fig, ax = plt.subplots(figsize=(17, 9))

    # 노란 막대
    ax.bar(
        labels,
        cagr_values,
        color='#FFC000',  # 노란색
        edgecolor='black',
        linewidth=1
    )

    # y축 격자선
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)

    # 0 기준선만 굵게 표시
    ax.axhline(0, color='black', linewidth=3)

    # x축 레이블 (아래로 충분히 내림)
    ax.set_xlabel('데이터 조합', fontsize=32, fontweight='bold', labelpad=40)
    ax.set_ylabel('Return Rate (%)', fontsize=32, fontweight='bold')

    # xticks: 수평 정렬, 중앙 정렬
    ax.set_xticks(labels)
    ax.set_xticklabels(labels, rotation=0, ha='center')

    # 레이아웃 및 아래·왼쪽 여백 확보
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, bottom=0.15)

    plt.show()

def my_graph():
    import matplotlib.pyplot as plt
    import numpy as np

    # ----------------------------------------------------
    # ✅ 글씨체 설정 (여기를 수정해 원하는 폰트로 변경하세요)
    # ----------------------------------------------------
    # '나눔고딕'이 설치되어 있다면 사용하고, 없다면 '맑은 고딕'을 사용합니다.
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['font.sans-serif'] = ['Malgun Gothic']
    except:
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['font.sans-serif'] = ['Malgun Gothic']

    plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 깨짐 방지

    # --- 기존 코드와 동일 ---
    # 1. 데이터 입력
    categories = ['Fin', 'Dis', 'Macro', 'Dis&Macro', 'Fin&Macro', 'Fin&Dis', 'Fin&Dis&Macro']
    return_rates = [-12.77, -4.21, -3.22, 4.99, 5.25, -5.56, 15.12]
    sharpe_ratios = [-0.4418, -0.1552, -0.0494, 0.1702, 0.1851, -0.1507, 0.42]
    max_drawdowns = [0.1006, 0.1401, 0.1356, 0.1765, 0.1727, 0.1066, 0.1332]

    # 2. 그래프 디자인 설정
    colors = ['pink', 'green','yellow','blue','orange','purple','red']
    bar_width = 0.8
    num_categories = len(categories)

    # 3. 그래프 생성 코드
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5)) # figsize 가로 길이 조절

    # --- 첫 번째 그래프: Return rate(%) ---
    ax1.bar(categories, return_rates, width=bar_width, color=colors, linewidth=1.5, edgecolor='black')
    ax1.set_ylabel('Return rate(%)', fontsize=16, fontweight='bold')
    ax1.set_ylim(min(return_rates) * 1.3, max(return_rates) * 1.3)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.axhline(0, color='grey', linewidth=0.8)
    ax1.set_xticks([])
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # --- 두 번째 그래프: Sharpe ratio ---
    ax2.bar(categories, sharpe_ratios, width=bar_width, color=colors, linewidth=1.5, edgecolor='black')
    ax2.set_ylabel('Sharpe ratio', fontsize=16, fontweight='bold')
    ax2.set_ylim(min(sharpe_ratios) * 1.3, max(sharpe_ratios) * 1.3)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.axhline(0, color='grey', linewidth=0.8)
    ax2.set_xticks([])
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # --- 세 번째 그래프: Maximum drawdown ---
    ax3.bar(categories, max_drawdowns, width=bar_width, color=colors, linewidth=1.5, edgecolor='black')
    ax3.set_ylabel('Maximum drawdown', fontsize=16, fontweight='bold')
    ax3.set_ylim(0, max(max_drawdowns) * 1.3)
    ax3.tick_params(axis='y', labelsize=12)
    ax3.set_xticks([])
    ax3.grid(axis='y', linestyle='--', alpha=0.7)

    # --- 범례 생성 ---
    # --- 범례 생성 (수정된 부분) ---
    legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], edgecolor='black', linewidth=1.5) for i in
                      range(num_categories)]

    # 첫 번째 줄 범례 (4개)
    fig.legend(legend_handles[:4], categories[:4], loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=14)
    # 두 번째 줄 범례 (3개)
    fig.legend(legend_handles[4:], categories[4:], loc='upper center', bbox_to_anchor=(0.5, 0.93), ncol=3, fontsize=14)


    plt.tight_layout(rect=[0, 0, 1, 0.85])
    plt.show()

# 함수 호출
my_graph()