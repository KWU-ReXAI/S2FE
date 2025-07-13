   if not predictions:
      return '0'

   # 다수결 투표 로직 (새로운 버전)
   # Counter를 사용하여 각 예측 값의 빈도를 계산
   counts = Counter(predictions)
   # 빈도수가 높은 순으로 정렬
   most_common = counts.most_common()

   # 1. 예측된 값이 하나뿐이거나, 최빈값이 명확히 하나일 경우 (동률이 아님)
   # most_common 리스트의 길이가 1이거나, 첫 번째 값의 빈도수가 두 번째 값의 빈도수보다 클 때
   if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
      # 가장 많이 나온 값을 그대로 반환
      return most_common[0][0]

   # 2. 최빈값이 동률일 경우
   else:
      # 동률인 값들을 저장할 리스트
      tied_values = []
      # 가장 높은 빈도수를 저장
      top_count = most_common[0][1]

      # most_common 리스트를 순회하며 가장 높은 빈도수와 같은 값을 모두 찾음
      for value, count in most_common:
         if count == top_count:
            # 계산을 위해 정수(int)로 변환하여 추가
            tied_values.append(int(value))
         else:
            # 정렬되어 있으므로, 더 낮은 빈도수가 나오면 중단
            break

      # 동률인 값들의 평균을 계산
      average = sum(tied_values) / len(tied_values)

      # 평균을 반올림 (0.5는 올림 처리)
      # Python의 round()는 0.5를 짝수에 가깝게 반올림하므로, 직접 구현합니다.
      # 예: 0.5 -> 1, -0.5 -> 0
      if average >= 0:
         rounded_average = int(average + 0.5)
      else:
         rounded_average = int(average - 0.5)

      # 최종 결과를 '+1', '0', '-1' 형식에 맞춰 문자열로 반환
      if rounded_average > 0:
         return f"+{rounded_average}"
      else:
         return str(rounded_average)