import subprocess

GIT = r"C:\Program Files\Git\cmd\git.exe"
subprocess.run([GIT, "add", "-u"])
subprocess.run([GIT, "commit", "--amend", "--no-edit"])
r = subprocess.run([GIT, "status"], capture_output=True, text=True)
print(r.stdout)
