import os 
import sys
import numpy as np

main_dir = "../DATASETS/IAM"
split = "train"

gtfile = f"{main_dir}/ascii/words.txt"

split_file_train = np.loadtxt(f"{main_dir}/splits/train.uttlist", dtype=str) 
split_file_val = np.loadtxt(f"{main_dir}/splits/validation.uttlist", dtype=str) 
split_file_test = np.loadtxt(f"{main_dir}/splits/test.uttlist", dtype=str) 

gt_words = {'train': [], 'val': [], 'test': []}

for line in open(gtfile):
    if not line.startswith("#"):
        info = line.strip().split()
        name = info[0]
        status = info[1]
        transcr = ' '.join(info[8:])

        if status == "ok":
            #print(status)

            name_parts = name.split('-')
            #pathlist = ['-'.join(name_parts[:i+1]) for i in range(len(name_parts))]
            split_tag = f"{name_parts[0]}-{name_parts[1]}"
            
            #print(name + " " + transcr + "\n")
            
            if split_tag in split_file_train:
                gt_words['train'].append(name + " " + transcr + "\n")
            if split_tag in split_file_val:
                gt_words['val'].append(name + " " + transcr + "\n")
            if split_tag in split_file_test:
                gt_words['test'].append(name + " " + transcr + "\n")       

for subset in gt_words.keys():
    print(subset)
    with open(os.path.join(f"gt_{subset}.txt"), "w") as f:
        f.writelines(gt_words[subset])