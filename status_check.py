import subprocess

GIT = r"C:\Program Files\Git\cmd\git.exe"
r = subprocess.run([GIT, "status"], capture_output=True, text=True)
print(r.stdout)
