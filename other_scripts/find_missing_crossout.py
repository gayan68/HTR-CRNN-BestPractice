import os 
import sys
import numpy as np
import pandas as pd
import shutil

main_dir = "../DATASETS/IAM"
gtfile = f"{main_dir}/ascii/words.txt"

split_file_train = np.loadtxt(f"{main_dir}/splits/train.uttlist", dtype=str) 
split_file_val = np.loadtxt(f"{main_dir}/splits/validation.uttlist", dtype=str) 
split_file_test = np.loadtxt(f"{main_dir}/splits/test.uttlist", dtype=str) 

striked_types = ["CROSS","DIAGONAL","DOUBLE_LINE","SCRATCH","SINGLE_LINE","WAVE","ZIG_ZAG","BLOT_1","BLOT_2"]
missing_file =[]

for struk_type in striked_types:

    for line in open(gtfile):
        if not line.startswith("#"):
            info = line.strip().split()
            name = info[0]
            status = info[1]
            transcr = ' '.join(info[8:])

            if status == "ok":
                name_parts = name.split('-')
                split_tag = f"{name_parts[0]}-{name_parts[1]}"
                
                if split_tag in split_file_train:
                    file_path = f"{main_dir}/striked_v2/train/images/{struk_type}/{name}.png"
                    if not os.path.exists(file_path):
                        missing_file.append([struk_type,"train",name])
                if split_tag in split_file_val:
                    file_path = f"{main_dir}/striked_v2/val/images/{struk_type}/{name}.png"
                    if not os.path.exists(file_path):
                        missing_file.append([struk_type,"val",name])
                if split_tag in split_file_test:
                    file_path = f"{main_dir}/striked_v2/test/images/{struk_type}/{name}.png"
                    if not os.path.exists(file_path):
                        missing_file.append([struk_type,"test",name])

for mfile in missing_file:
    word_file_src = f"{main_dir}/processed_words/{mfile[1]}/{mfile[2]}.png"
    #word_file_dst = f"./error_files/{mfile[2]}.png"
    word_file_dst = f"{main_dir}/striked_v2/{mfile[1]}/images/{mfile[0]}/{mfile[2]}.png"
    print(word_file_dst)
    shutil.copy(word_file_src, word_file_dst)


df = pd.DataFrame(missing_file, columns=["struck_type", "split" , "file_name"])
df.to_csv("missing_struck.csv", index=False)

#print(missing_file)