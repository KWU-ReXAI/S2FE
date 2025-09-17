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
    cagr_selection = [13.92, -8.14, -8.46, -8.97, -9.22]

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
    cagr_features = [-13.42, 5.00, 6.11, 13.92, -2.79]

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


import matplotlib.pyplot as plt
import numpy as np

def my_graph():
    # ----------------------------------------------------
    # 글씨체 설정
    # ----------------------------------------------------
    try:
        plt.rcParams['font.family'] = 'arial'
    except RuntimeError:
        plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 데이터 입력
    #categories = ['Fin', 'Fin&Dis', 'Dis', 'Macro', 'Dis&Macro', 'Fin&Macro', 'Fin&Dis&Macro']
    categories = ['All-D&M', 'All-M', 'All-F&M', 'All-F&D', 'All-F', 'All-D', 'All']
    return_rates = [-12.77, -5.56, -4.21, -3.22,  4.99, 5.25, 16.23]
    sharpe_ratios = [-0.4418, -0.1507,  -0.1552, -0.0494, 0.1702, 0.1851, 0.42]
    max_drawdowns = [0.1006,  0.1066, 0.1401, 0.1356, 0.1765, 0.1727, 0.1332]

    # 2. 그래프 디자인 설정
    patterns = ['///', '--', 'xxx', '**', '++', 'oo', '\\\\']
    colors = ['pink', 'gray', 'yellow', 'blue', 'orange', 'purple', 'green']
    bar_width = 0.8
    num_categories = len(categories)
    plt.rcParams['hatch.linewidth'] = 1.5

    # 3. 그래프 생성 코드
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # --- 첫 번째 그래프: Return rate(%) ---
    for i in range(num_categories):
        ax1.bar(categories[i], return_rates[i], width=bar_width, color=colors[i],
                linewidth=1.5, edgecolor='black', hatch=patterns[i], zorder=3)
    ax1.set_ylabel('Return rate(%)', fontsize=30, fontweight='bold', labelpad=10)
    # ✅ y축 범위를 살짝 확장하여 맨 아래/위 선이 보이도록 합니다.
    ax1.set_ylim(-22, 22)
    # ✅ y축 눈금(tick) 색상을 회색으로 변경합니다.
    ax1.tick_params(axis='y', labelsize=24, color='grey')
    ax1.set_yticks([-20,-15, -10,-5, 0, 5,10, 15,20])
    ax1.axhline(0, color='grey', linewidth=1.2,zorder=0)
    ax1.set_xticks([])
    ax1.grid(axis='y', linestyle='-', color='grey', alpha=0.7,zorder=0, linewidth=1.2)

    # --- 두 번째 그래프: Sharpe ratio ---
    for i in range(num_categories):
        ax2.bar(categories[i], sharpe_ratios[i], width=bar_width, color=colors[i],
                linewidth=1.5, edgecolor='black', hatch=patterns[i], zorder=3)
    ax2.set_ylabel('Sharpe ratio', fontsize=30, fontweight='bold', labelpad=10)
    ax2.set_ylim(-0.55, 0.55)
    ax2.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax2.tick_params(axis='y', labelsize=24, color='grey')
    ax2.axhline(0, color='grey', linewidth=0.8, zorder=0)
    ax2.set_xticks([])
    ax2.grid(axis='y', linestyle='-', color='grey', alpha=0.7, zorder=0, linewidth=1.2)

    # --- 세 번째 그래프: Maximum drawdown ---
    for i in range(num_categories):
        ax3.bar(categories[i], max_drawdowns[i], width=bar_width, color=colors[i],
                linewidth=1.5, edgecolor='black', hatch=patterns[i], zorder=3)
    ax3.set_ylabel('Maximum drawdown', fontsize=30, fontweight='bold', labelpad=10)
    # ✅ 0선이 잘리지 않도록 y축 최소값을 살짝 내립니다.
    ax3.set_ylim(-0.01, 0.22)
    ax3.set_yticks([0, 0.05, 0.1, 0.15, 0.2])
    ax3.tick_params(axis='y', labelsize=24, color='grey')
    ax3.set_xticks([])
    ax3.grid(axis='y', linestyle='-', color='grey', alpha=0.7, zorder=0, linewidth=1.2)

    # --- 공통 스타일 수정 ---
    for ax in [ax1, ax2, ax3]:
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
        for spine in ax.spines.values():
            spine.set_visible(False)

    # --- 범례 생성 ---
    legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], edgecolor='black',
                                    linewidth=1.5, hatch=patterns[i]) for i in range(num_categories)]

    fig.legend(legend_handles[:], categories[:], loc='upper center', bbox_to_anchor=(0.5, 1), ncol=7,
               prop={'weight': 'bold', 'size': 25}, columnspacing=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.show()



def generate_cagr_chart():
    # ----------------------------------------------------
    # 글씨체 설정
    # ----------------------------------------------------
    try:
        plt.rcParams['font.family'] = 'arial'
    except RuntimeError:
        plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

    # 데이터
    labels = ['WMR', 'Agg3', 'B&H', 'RF', 'ERG', 'S3CE', 'SEFA']
    cagr_values = [-8.19, -7.10, -5.20, -3.88, -1.47, 1.69, 13.92]

    # 그래프 설정
    plt.figure(figsize=(7, 5))
    plt.bar(labels, cagr_values, color='#FFD700', edgecolor='black', linewidth=1.2, zorder=3)

    plt.axhline(y=0, color='grey', linestyle='-', linewidth=1.2, zorder=0)

    # y축 레이블 및 범위 설정
    plt.ylabel('CAGR (%)', fontsize=20, fontweight='bold')
    plt.yticks(np.arange(-10, 20, 5), fontsize=18, fontweight='bold')
    plt.ylim(min(cagr_values) * 1.3, max(cagr_values) * 1.3)

    # x축 레이블 설정
    plt.xticks(fontsize=18, fontweight='bold')

    plt.grid(axis='y', linestyle='-', color='grey', alpha=0.7, linewidth=1.2, zorder=0)

    # --- 스타일 추가 ---
    ax = plt.gca()

    # 그래프 테두리 제거
    for spine in ax.spines.values():
        spine.set_visible(False)

    # y축 눈금(tick) 색상을 회색으로 변경
    ax.tick_params(axis='y', color='grey')

    # ✅ x축 눈금(tick)의 길이를 다시 0으로 만들어 보이지 않게 합니다.
    ax.tick_params(axis='x', length=0)
    # --------------------

    # 레이아웃 조정
    plt.tight_layout()

    # 그래프 보여주기
    plt.show()


import matplotlib.pyplot as plt
import numpy as np

def generate_llm_chart():
    # ----------------------------------------------------
    # 글씨체 설정
    # ----------------------------------------------------
    try:
        plt.rcParams['font.family'] = 'arial'
    except RuntimeError:
        plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

    # 데이터
    labels = ["w/o LLM", "Vid & Art", "Video", "Vid | Art", "Article"]
    cagr_values = [7.1, 9.0, 9.3, 11.7, 18.1]

    # 그래프 설정
    plt.figure(figsize=(8, 5))
    plt.bar(labels, cagr_values, color='#FFD700', edgecolor='black', linewidth=1.2, zorder=3,width=0.5)

    plt.axhline(y=0, color='grey', linestyle='-', linewidth=1.2, zorder=0)

    # y축 레이블 및 범위 설정
    plt.ylabel('CAGR (%)', fontsize=24, fontweight='bold')
    # ✅ y축 눈금을 0부터 시작하도록 수정
    plt.yticks(np.arange(0, 26, 5), fontsize=24, fontweight='bold')
    # ✅ y축 시작점을 0으로 수정
    plt.ylim(0, max(cagr_values) * 1.3)

    # x축 레이블 설정
    plt.xticks(fontsize=20, fontweight='bold')

    plt.grid(axis='y', linestyle='-', color='grey', alpha=0.7, linewidth=1.2, zorder=0)

    # --- 스타일 추가 ---
    ax = plt.gca()

    # 그래프 테두리 제거
    for spine in ax.spines.values():
        spine.set_visible(False)

    # y축 눈금(tick) 색상을 회색으로 변경
    ax.tick_params(axis='y', color='grey')

    # x축 눈금(tick)의 길이를 0으로 만들어 보이지 않게 합니다.
    ax.tick_params(axis='x', length=0)
    # --------------------

    # 레이아웃 조정
    plt.tight_layout()

    # 그래프 보여주기
    plt.show()

if __name__ == '__main__':
    my_graph()
    generate_llm_chart()