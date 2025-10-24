import matplotlib.pyplot as plt
import numpy as np

def cagr(): # 특징 선택 개수 & 종목 선택 비율 실험 Figure
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
    plt.xticks(selection_ratios,fontweight='bold')
    plt.ylim(-15, 20)
    plt.yticks(list(range(-15, 21, 5)),fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("./figure_file/그림9.png")

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
    plt.xticks(feature_counts,fontweight='bold')
    plt.ylim(-15, 20)
    plt.yticks(list(range(-15, 21, 5)),fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("./figure_file/그림10.png")


def my_graph(): # 절제 연구 실험 Figure
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
    plt.savefig("./figure_file/그림8.png")



def generate_cagr_chart(): # 비교 모델 실험 CAGR 그래프
    # ----------------------------------------------------
    # 글씨체 설정
    # ----------------------------------------------------
    try:
        plt.rcParams['font.family'] = 'arial'
    except RuntimeError:
        plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

    # 데이터
    labels = ['WMR', 'Agg3', 'B&H', 'RF', 'ERG', 'S3CE', 'S2FE']
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
    plt.savefig("./figure_file/그림6.png")



def generate_llm_chart(): # LLM CAGR 그래프
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
    plt.savefig("./figure_file/그림7.png")

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def plot_kospi200_chart(csv_path: str): # 실험 테이터셋(실험 기간) Figure: figure.pptx에서 추가 편집하였습니다.
    """
    주어진 CSV 파일 경로를 사용하여 KOSPI 200 지수 차트를 생성하고 표시합니다.
    x축은 매년 4분기 시작일을 기준으로 '{연도}_Q4' 형식으로 표시됩니다.
    'KS200.csv' 파일의 '날짜', '종가' 컬럼명에 맞춰 수정되었으며, 범례가 제거되었습니다.
    x축 범위는 2015년 4분기부터 2024년 4분기까지로 제한됩니다.
    x축/y축 라벨과 눈금 값 모두 크고 굵게 표시됩니다.

    Args:
        csv_path (str): KOSPI 200 데이터가 포함된 CSV 파일의 경로.
    """
    try:
        # 한글 폰트 설정
        try:
            plt.rc('font', family='Malgun Gothic')  # Windows
        except:
            try:
                plt.rc('font', family='AppleGothic')  # macOS
            except:
                font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
                if fm.findfont(fm.FontProperties(fname=font_path)):
                    plt.rc('font', family=fm.FontProperties(fname=font_path).get_name())
        plt.rcParams['axes.unicode_minus'] = False

        # 1. CSV 파일 읽기 및 데이터 필터링
        df = pd.read_csv(csv_path, parse_dates=['날짜'], index_col='날짜')
        df.sort_index(inplace=True)
        start_date = '2015-10-01'
        end_date = '2024-10-01'
        df_filtered = df.loc[start_date:end_date].copy()

        if df_filtered.empty:
            print(f"오류: {start_date}부터 {end_date}까지의 데이터가 파일에 없습니다.")
            return

        price_column = '종가'
        if price_column not in df_filtered.columns:
            raise ValueError(f"차트를 그릴 가격 데이터 컬럼('{price_column}')을 찾을 수 없습니다.")

        # 2. 그래프 생성
        fig, ax = plt.subplots(figsize=(17, 4))  # 세로 길이를 조금 늘려 공간 확보
        ax.plot(df_filtered.index, df_filtered[price_column], color='royalblue',linewidth=2.5)

        # 3. x축 눈금 및 레이블 설정
        years = range(2015, 2025)
        xticks = [pd.Timestamp(f'{year}-10-01') for year in years]
        xtick_labels = [f'{year-2000}/10' for year in years]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, ha='center',fontsize=6)

        # 4. x축 범위 명시적 설정
        ax.set_xlim(pd.Timestamp(start_date), pd.Timestamp(end_date))
        ax.set_ylim(160,450)

        # 5. 그래프 스타일 및 정보 추가
        #ax.set_title('KOSPI 200 지수 (2015_Q4 - 2024_Q4)', fontsize=18, fontweight='bold', pad=20)
        ax.set_ylabel('KOSPI200 Index', fontsize=20, fontweight='bold')
        ax.set_xlabel('Date', fontsize=20, fontweight='bold')

        # 6. x축과 y축 눈금 값 스타일 변경 (수정된 부분)
        # 글씨 크기를 12로, 굵게(bold) 설정
        plt.setp(ax.get_xticklabels(), fontsize=17, fontweight='bold')
        plt.setp(ax.get_yticklabels(), fontsize=17, fontweight='bold')


        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        # 'color'를 'facecolor'로 변경하여 면과 테두리 색을 분리
        ax.axvspan('2020-10-01', '2021-10-01', facecolor='skyblue', alpha=0.3, edgecolor='black', linewidth=3, zorder=-4)
        ax.axvspan('2021-10-01', '2022-10-01', facecolor='lightgreen', alpha=0.3, edgecolor='black', linewidth=3,
                   zorder=-3)
        ax.axvspan('2022-10-01', '2023-10-01', facecolor='yellow', alpha=0.3, edgecolor='black', linewidth=3, zorder=-2)
        ax.axvspan('2023-10-01', '2024-10-01', facecolor='lightpink', alpha=0.3, edgecolor='black', linewidth=3,
                   zorder=-1)
        # y축 400~450 & x축 2016-10-01 ~ 2020-10-01 영역을 투명 회색으로 표시
        ax.fill_between([pd.to_datetime('2015-10-01'), pd.to_datetime('2020-10-01')], 400, 450, color='lightgray', alpha=1,
                        zorder=-3.5)
        ax.fill_between([pd.to_datetime('2016-10-01'), pd.to_datetime('2021-10-01')], 350, 400, color='lightgray', alpha=1,
                        zorder=-2.5)
        ax.fill_between([pd.to_datetime('2017-10-01'), pd.to_datetime('2022-10-01')], 300, 350, color='lightgray',
                        alpha=1,
                        zorder=-1.5)
        ax.fill_between([pd.to_datetime('2018-10-01'), pd.to_datetime('2023-10-01')], 250, 300, color='lightgray',
                        alpha=1,
                        zorder=-0.5)

        ax.grid(True, linestyle='-', alpha=0.6)

        # 레이아웃을 조정하여 라벨이 잘리지 않도록 합니다.
        plt.tight_layout()
        plt.savefig("./figure_file/그림5_PPT적용전.png")

    except FileNotFoundError:
        print(f"오류: '{csv_path}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    except KeyError as e:
        print(f"오류: CSV 파일에서 필요한 컬럼({e})을 찾을 수 없습니다. 파일 내용을 확인해주세요.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

if __name__ == '__main__':
    #cagr()
    #my_graph()
    #generate_llm_chart()
    generate_cagr_chart()
    #plot_kospi200_chart("./KS200.csv")