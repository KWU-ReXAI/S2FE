import subprocess

try:
    subprocess.run(f"python datapreprocessing.py --use_all True --isall cluster", shell=True, check=True)
    subprocess.run(f"python clustering.py", shell=True, check=True)
    subprocess.run(f"python datapreprocessing.py --use_all True --isall False", shell=True, check=True)
    subprocess.run(f"python datapreprocessing.py --use_all True --isall True", shell=True, check=True)
    subprocess.run(f"python train.py --testNum 20", shell=True, check=True)
    subprocess.run(f"python test.py --testNum 20", shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"명령어 실행 중 오류가 발생했습니다: {e}")