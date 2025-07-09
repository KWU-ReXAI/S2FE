import subprocess

try:
    # check=True를 추가하여 오류 발생 시 예외를 발생시킴
    subprocess.run(f"python datapreprocessing.py --isall True --public True --macroeconomic True", shell=True,
                   check=True)
    subprocess.run(f"python datapreprocessing.py --isall cluster --public True --macroeconomic True", shell=True,
                   check=True)
    subprocess.run(f"python clustering.py", shell=True, check=True)
    subprocess.run(f"python datapreprocessing.py --isall False --public True --macroeconomic True", shell=True,
                   check=True)
    subprocess.run(f"python train.py --testNum 10 --data Public_Macroeconomic", shell=True, check=True)
    subprocess.run(f"python test.py --testNum 10 --data Public_Macroeconomic", shell=True, check=True)

    #####################################################################################################################

    subprocess.run(f"python datapreprocessing.py --isall True --public True --financial True", shell=True,
                   check=True)
    subprocess.run(f"python datapreprocessing.py --isall cluster --public True --financial True", shell=True,
                   check=True)
    subprocess.run(f"python clustering.py", shell=True, check=True)
    subprocess.run(f"python datapreprocessing.py --isall False --public True --financial True", shell=True,
                   check=True)
    subprocess.run(f"python train.py --testNum 10 --data Financial_Public", shell=True, check=True)
    subprocess.run(f"python test.py --testNum 10 --data Financial_Public", shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"명령어 실행 중 오류가 발생했습니다: {e}")