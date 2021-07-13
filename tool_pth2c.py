import os
import sys
import shutil
import numpy as np

if len(sys.argv)!=2:
    print("Insert session folder (output/train/<session_folder>) as argument.\n")
    exit(-1)

folder = sys.argv[1] # NLRAV / NSVFQ
session_path = "output/train/"+folder+"/"
if os.path.exists(session_path) == False:
    print(session_path+" does not exist!")
    exit()
session_path_parameters = session_path+"parameters/"
if os.path.exists(session_path_parameters) == False:
    print(session_path+" does not exist!")
    exit()

file_list = os.listdir(session_path_parameters)
file_list.sort()



model_parameters_path = "output/model_parameters/"+folder+"/"

try:
    os.mkdir(model_parameters_path)
except OSError:
    print("Error in session creation ("+model_parameters_path+"), session already exists?")
    print("Overwrite the session? (y/n) ", end='')
    force_write = input()
    if force_write == "y":
        try:
            shutil.rmtree(model_parameters_path)
            os.mkdir(model_parameters_path)
        except OSError:
            print("Error in session creation ("+model_parameters_path+").")
            exit()
    else:
        exit()

np.set_printoptions(threshold=sys.maxsize)

f = open(model_parameters_path+"/model_parameters.h", "w")

for file in file_list:
    if file.endswith(".txt"):
        x = np.loadtxt(os.path.join(session_path_parameters, file)).flatten()
        # print(str(os.path.join(session_path_parameters, file))+"\nSize: "+str(x.size)+"\n"+str(x)+"\n")
        f.write("#define "+str(str(os.path.join(session_path_parameters, file).replace(session_path_parameters,"")).upper().split( "." )[0])+" ")
        f.write(str(x[0]))
        for line in x[1:]:
            f.write(", ")
            f.write(str(line))
        f.write("\n\n")
    if file.endswith(".npy"):
        x = np.load(os.path.join(session_path_parameters, file)).flatten()
        # print(str(os.path.join(session_path_parameters, file))+"\nSize: "+str(x.size)+"\n"+str(x)+"\n")
        f.write("#define "+str(str(os.path.join(session_path_parameters, file).replace(session_path_parameters,"")).upper().split( "." )[0])+"_SIZE "+str(x.size)+"\n")
        f.write("#define "+str(str(os.path.join(session_path_parameters, file).replace(session_path_parameters,"")).upper().split( "." )[0])+" {")
        f.write(str(int(x[0])))
        for line in x[1:]:
            f.write(", ")
            f.write(str(int(line)))
        f.write("}\n\n")

f.close()
