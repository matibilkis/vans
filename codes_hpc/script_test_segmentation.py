import os

name = "due"
if not os.path.exists(name):
    os.makedirs(name)
    with open(name+"/number_run.txt", "w+") as f:
        f.write("0")
        f.close()
    number_run = 0
else:
    with open(name+"/number_run.txt", "r") as f:
        a = f.readlines()[0]
        f.close()
    with open(name+"/number_run.txt", "w") as f:
        f.truncate(0)
        number_run = int(a)+1
        f.write(str(int(a)+1))
        f.close()

dir_to_save = name+"/run_"+str(number_run)
os.makedirs(dir_to_save)
os.makedirs(dir_to_save+"/data_collected")
