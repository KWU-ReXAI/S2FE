import subprocess

subprocess.run(f"python datapreprocessing.py --isall True", shell=True)
subprocess.run(f"python datapreprocessing.py --isall cluster", shell=True)
subprocess.run(f"python clustering.py", shell=True)
subprocess.run(f"python datapreprocessing.py --isall False", shell=True)
subprocess.run(f"python train.py --testNum 5", shell=True)
subprocess.run(f"python test.py --testNum 5", shell=True)