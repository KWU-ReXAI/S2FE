# 필요한 라이브러리 임포트
import os  # 파일 및 디렉터리 경로 처리를 위한 표준 라이브러리
import logging  # 로깅 설정을 위한 라이브러리
import torch  # PyTorch 프레임워크로, 모델 로딩과 연산에 사용됨
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration  # 토크나이저 및 요약 모델 로딩

# transformers 라이브러리의 불필요한 경고 메시지를 억제하기 위해 로깅 레벨을 ERROR로 설정
logging.getLogger("transformers").setLevel(logging.ERROR)

# 모델과 토크나이저를 불러오겠다는 안내 출력
print("모델과 토크나이저를 로드하는 중 (KoBART-summarization)...")

# 사전 학습된 한국어 요약 모델에 맞는 토크나이저 로드 (KoBART용 FastTokenizer)
tokenizer = PreTrainedTokenizerFast.from_pretrained("digit82/kobart-summarization")

# 사전 학습된 KoBART 요약 모델 로드
model = BartForConditionalGeneration.from_pretrained("digit82/kobart-summarization")

# 평가 모드로 설정하여 dropout 등의 학습 관련 기능 비활성화
model.eval()

# 현재 시스템에서 CUDA(GPU)가 사용 가능한지 확인하고, 가능하면 'cuda'를 사용, 그렇지 않으면 'cpu'로 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 모델을 선택한 디바이스로 이동 (GPU 가속 또는 CPU 실행용)
model.to(device)


##############################################
# 지정한 텍스트 파일을 요약하고 결과를 저장하는 함수 정의
##############################################
def summarize_file_simple(
    input_path: str,  # 입력 텍스트(.txt) 파일 경로
    output_path: str,  # 요약 결과를 저장할 .txt 파일 경로
    max_input_tokens: int = 1024,  # 모델이 처리할 수 있는 최대 입력 토큰 수
    summary_max_length: int = 64,  # 생성할 요약의 최대 토큰 수
    num_beams: int = 4,  # beam search를 위한 빔의 개수 (탐색 폭 조절)
    device: str = "cpu",  # 사용할 연산 디바이스 (cpu 또는 cuda)
) -> str:
    """
    입력 텍스트 파일을 요약하고 결과를 저장하며, 생성된 요약 문자열을 반환하는 함수
    """

    # 지정한 입력 파일 경로의 내용을 읽어오기
    print(f"\n원본 파일을 읽는 중: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        article_text = f.read().strip()  # 파일에서 전체 텍스트를 읽고 양쪽 공백 제거

    # 만약 읽어온 텍스트가 비어 있다면 경고 출력 후 빈 문자열 반환
    if not article_text:
        print("  → 입력 파일이 비어 있습니다. 파일 경로를 확인하세요.")
        return ""

    # 요약 전에 원본 텍스트 일부(앞 200자) 및 전체 길이 출력
    print("=== 원본 텍스트 샘플 (200자) ===")
    print(article_text[:200].replace("\n", " "))  # 개행문자를 공백으로 바꿔 출력
    print(f"원본 텍스트 길이: {len(article_text)}자")
    print("=============================")

    # 토크나이저를 이용해 원본 텍스트를 토큰 ID 리스트로 인코딩 (special token 제외)
    all_token_ids = tokenizer.encode(article_text, add_special_tokens=False)

    # 토큰 수가 최대 입력 토큰 수를 초과할 경우 자르기
    if len(all_token_ids) > max_input_tokens:
        input_token_ids = all_token_ids[:max_input_tokens]  # 앞쪽 max_input_tokens 만큼 사용
        truncated = True  # 자름 여부 기록
    else:
        input_token_ids = all_token_ids  # 그대로 사용
        truncated = False  # 자르지 않음

    # 자른 토큰 ID를 다시 텍스트로 디코딩 (요약 입력으로 사용)
    truncated_text = tokenizer.decode(input_token_ids, skip_special_tokens=True)

    # 잘랐다면 경고 메시지 출력
    if truncated:
        print(f"  → 입력 토큰이 {len(all_token_ids)}개여서, 앞 {max_input_tokens}개로 잘라냈습니다.")

    # 토크나이저로 텍스트를 인코딩하고 텐서 형태로 변환 (모델 입력 준비)
    model.to(device)  # 혹시 모델이 다른 디바이스에 있을 수 있으므로 재지정
    inputs = tokenizer.encode_plus(
        truncated_text,  # 입력 텍스트
        return_tensors="pt",  # PyTorch 텐서로 반환
        max_length=max_input_tokens,  # 최대 길이 지정
        truncation=True,  # 길이가 초과하면 자동 자르기
    )
    input_ids = inputs["input_ids"].to(device)  # 입력 ID를 디바이스로 이동
    attention_mask = inputs["attention_mask"].to(device)  # attention mask도 이동

    # 요약 생성: gradient 계산 없이 추론만 수행
    with torch.no_grad():
        summary_ids = model.generate(
            input_ids,  # 모델 입력
            attention_mask=attention_mask,  # 입력 마스크
            max_length=summary_max_length,  # 생성할 최대 토큰 수
            num_beams=num_beams,  # beam search 탐색 폭
            early_stopping=True,  # 충분히 좋은 요약이 나오면 일찍 멈춤
        )

    # 생성된 요약 토큰 ID를 다시 텍스트로 디코딩하고 좌우 공백 제거
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

    # 출력 경로의 디렉토리가 존재하지 않으면 생성
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 요약 텍스트를 output_path 경로에 저장
    print(f"요약 결과를 저장 중: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write(summary)

    # 처리 완료 메시지 출력 및 결과 요약 반환
    print("요약 완료!")
    return summary


##############################################
# 실제 실행되는 메인 루틴 정의
##############################################
if __name__ == "__main__":

    # 첫 번째 카테고리: 산업재 섹터에 해당하는 기업 코드 리스트
    code_list_1 = [
        '000120', '000150', '001120', '003490', '003570', '006260',
        '010120', '011200', '025540', '047050', '047810', '051600', '086280'
    ]
    base_input_dir_1 = "./script/산업재"  # 산업재 입력 텍스트 파일들이 있는 디렉터리 경로
    base_output_dir_1 = "./summary_video/산업재"  # 요약 결과를 저장할 디렉터리 경로

    # 산업재 카테고리의 각 기업별로 요약 수행
    for code in code_list_1:
        input_dir = os.path.join(base_input_dir_1, code)  # 기업별 입력 경로 생성
        output_dir = os.path.join(base_output_dir_1, code)  # 기업별 출력 경로 생성

        # 입력 디렉터리가 없을 경우 경고 출력 후 건너뛰기
        if not os.path.isdir(input_dir):
            print(f"입력 경로가 존재하지 않습니다: {input_dir}")
            continue

        # 출력 디렉터리가 없다면 생성
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 입력 디렉터리의 모든 파일에 대해 반복
        for filename in os.listdir(input_dir):
            if not filename.endswith(".txt"):  # .txt 파일만 처리
                continue

            input_path = os.path.join(input_dir, filename)  # 입력 파일 경로
            output_path = os.path.join(output_dir, filename)  # 출력 파일 경로

            # 요약 함수 호출
            summarize_file_simple(
                input_path,
                output_path,
                max_input_tokens=1024,
                summary_max_length=64,
                num_beams=4,
                device=device,
            )

    # 두 번째 카테고리: 정보기술 섹터에 해당하는 기업 코드 리스트
    code_list_2 = [
        '000660', '003550', '004710', '005930', '006400', '008060',
        '009150', '011070', '018260', '020150', '029530', '034220',
        '034730', '042700', '066570'
    ]
    base_input_dir_2 = "./script/정보기술"  # 정보기술 입력 텍스트 디렉터리
    base_output_dir_2 = "./summary_video/정보기술"  # 정보기술 요약 저장 디렉터리

    # 정보기술 카테고리 기업별 요약 반복 처리
    for code in code_list_2:
        input_dir = os.path.join(base_input_dir_2, code)  # 입력 경로
        output_dir = os.path.join(base_output_dir_2, code)  # 출력 경로

        # 입력 경로가 존재하지 않으면 경고 후 건너뜀
        if not os.path.isdir(input_dir):
            print(f"입력 경로가 존재하지 않습니다: {input_dir}")
            continue

        # 출력 디렉터리가 없으면 생성
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 해당 디렉터리 내 모든 텍스트 파일 처리
        for filename in os.listdir(input_dir):
            if not filename.endswith(".txt"):  # .txt 파일만 처리
                continue

            input_path = os.path.join(input_dir, filename)  # 입력 경로
            output_path = os.path.join(output_dir, filename)  # 출력 경로

            # 요약 함수 호출
            summarize_file_simple(
                input_path,
                output_path,
                max_input_tokens=1024,
                summary_max_length=64,
                num_beams=4,
                device=device,
            )

    # 전체 요약 프로세스 종료 메시지
    print("\n모든 요약 작업이 완료되었습니다.")
