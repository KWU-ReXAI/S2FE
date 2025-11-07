import pandas as pd
import os


def find_top_3_cagr_nums():
    """
    1부터 10까지의 num을 순회하며 지정된 경로의 CSV 파일에서
    'CAGR' 행 'Average' 열 값이 가장 큰 상위 3개의 num과 그 값을 찾습니다.
    """

    # 기본 경로 설정
    base_path = "./result_original/result_S2FE_SectorAll"

    # 결과를 저장할 리스트
    # (CAGR값, num) 튜플 형태로 저장하여 나중에 정렬하기 쉽게 합니다.
    results_list = []

    print("파일 순회를 시작합니다...")

    # 1부터 10까지 (11 미만) 순회
    for num in range(1, 11):
        try:
            # 동적으로 파일 경로 생성
            file_name = f"train_result_file_{num}.csv"
            dir_name = f"train_result_dir_{num}"
            file_path = os.path.join(base_path, dir_name, file_name)

            # CSV 파일 읽기 (index_col=0 옵션 사용)
            df = pd.read_csv(file_path, index_col=0)

            # 'CAGR' 행의 'Average' 열 값 가져오기
            current_cagr_average = df.loc['CAGR', 'Average']

            # (값, num) 튜플로 리스트에 추가
            results_list.append((current_cagr_average, num))

        except FileNotFoundError:
            print(f"경고: {num}번 파일({file_path})을 찾을 수 없습니다. 건너뜁니다.")
        except KeyError:
            print(f"경고: {num}번 파일({file_path})에 'CAGR' 또는 'Average' 데이터가 없습니다. 건너뜁니다.")
        except Exception as e:
            print(f"오류: {num}번 파일({file_path}) 처리 중 오류 발생: {e}")

    # --- 모든 파일 순회 완료 ---

    if not results_list:  # results_list가 비어있다면
        print("\n--- 최종 결과 ---")
        print("유효한 데이터를 가진 파일을 하나도 찾지 못했습니다.")
        return []

    # 리스트를 CAGR 값(튜플의 첫 번째 요소) 기준으로 내림차순 정렬
    # (값이 큰 것이 맨 앞으로 오도록)
    results_list.sort(key=lambda item: item[0], reverse=True)

    # 상위 3개 추출
    # (만약 파일이 2개밖에 없었다면, [ :3 ] 슬라이싱은 오류 없이 2개만 반환합니다)
    top_3_results = results_list[:3]

    # 최종 결과 출력
    print("\n--- 최종 결과 (상위 3개) ---")

    if not top_3_results:
        print("결과를 찾지 못했습니다.")
    else:
        for i, (cagr_value, num) in enumerate(top_3_results):
            # i는 0부터 시작하므로 순위는 i+1
            print(f"순위 {i + 1}: num = {num} (CAGR: {cagr_value})")

    return top_3_results


import pandas as pd
import os


def find_top_3_test_cagr_average():
    """
    test_result_file.csv를 읽어 'CAGR' 행에 대해
    'Test {num}' 그룹별(p1~p4) 평균을 계산하고,
    그 평균이 가장 높은 'Test {num}' 그룹 상위 3개를 찾습니다.
    """

    # 1. 파일 경로 정의
    file_path = "result/result_S2FE_SectorAll/test_result_dir/test_result_file.csv"

    print(f"파일 분석 시작: {file_path}")

    try:
        # 2. CSV 파일 읽기 (MultiIndex 헤더)
        df = pd.read_csv(file_path, header=[0, 1], index_col=0)

        # 3. 'CAGR' 행 데이터 추출
        cagr_row = df.loc['CAGR']

        # 4. 그룹별 평균 계산
        all_group_averages = cagr_row.groupby(level=0).mean()

        # 5. 'Test '로 시작하는 그룹만 필터링
        test_averages = all_group_averages[all_group_averages.index.str.startswith('Test ')]

        if test_averages.empty:
            print(f"오류: {file_path}에서 'Test '로 시작하는 열 그룹을 찾을 수 없습니다.")
            return None

        # 6. 평균값을 기준으로 '내림차순 정렬'
        sorted_test_averages = test_averages.sort_values(ascending=False)

        # 7. 정렬된 결과에서 상위 3개 항목 선택
        # .head(3)는 항목이 3개 미만(예: 2개)이어도 오류 없이 2개만 반환합니다.
        top_3_results = sorted_test_averages.head(3)

        # 8. 결과 출력
        print("\n--- 최종 결과 (상위 3개) ---")

        if top_3_results.empty:
            print("데이터를 찾을 수 없습니다.")
        else:
            rank = 1
            # top_3_results는 인덱스(Test 5 등)와 값(CAGR 평균)을 가진 Series입니다.
            # .items()를 사용해 (인덱스, 값) 쌍으로 순회합니다.
            for test_name, avg_value in top_3_results.items():
                print(f"순위 {rank}: {test_name} (평균 CAGR: {avg_value})")
                rank += 1

        # Series 객체 자체를 반환
        return top_3_results

    except FileNotFoundError:
        print(f"오류: 파일({file_path})을 찾을 수 없습니다.")
        return None
    except KeyError:
        print(f"오류: 파일({file_path})에 'CAGR' 행이 없습니다.")
        return None
    except Exception as e:
        print(f"파일 처리 중 예기치 못한 오류 발생: {e}")
        return None


# 함수 실행
if __name__ == "__main__":
    find_top_3_cagr_nums()
    top_3 = find_top_3_test_cagr_average()