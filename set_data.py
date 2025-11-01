import subprocess

try:
    subprocess.run(f"python get_price.py", shell=True, check=True)
    subprocess.run(f"python kospi200.py", shell=True, check=True)
    subprocess.run(f"python get_koreanfeature.py", shell=True, check=True)
    subprocess.run(f"python seperate_files.py", shell=True, check=True)
    subprocess.run(f"python macro_economic.py", shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"명령어 실행 중 오류가 발생했습니다: {e}")