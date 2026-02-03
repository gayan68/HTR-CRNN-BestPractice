import argparse
from omegaconf import OmegaConf

import sys
import os
import re
import pickle
import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.htr_dataset2 import HTRDatasetBoth
import pandas as pd

from models import HTRNet
from utils.metrics import CER, WER

class HTREval(nn.Module):
    def __init__(self, config):
        super(HTREval, self).__init__()
        self.config = config

        self.prepare_dataloaders()
        self.prepare_net()

    def prepare_dataloaders(self):

        config = self.config

        # prepare datset loader
        dataset_folder1 = config.data.clean_path
        dataset_folder2 = config.data.strike_path
        fixed_size = (config.preproc.image_height, config.preproc.image_width)

        config.probability.clean = config_probability2_clean
        config.probability.striked_types = config_probability2_striked_types
       

        print(f"config.probability.clean: {config.probability.clean}")
        print(f"config.probability.striked_types: {config.probability.striked_types}")

        # To get the number of classes idetified in thetreining ##
        config.data.dataset = "CLEAN"
        train_set = HTRDatasetBoth(config, 'train', fixed_size=fixed_size, transforms=None)
        classes = train_set.character_classes

        classes += ' ' 
        classes = np.unique(classes)

        print(f"############## Calsses: {len(classes)}")
        print(classes)
        #########################################################
        config.data.dataset = test_dataset

        test_set = HTRDatasetBoth(config, 'test', fixed_size=fixed_size, transforms=None, character_classes=classes)
        print('# testing lines ' + str(test_set.__len__()))

        # load classes from the training set saved in the data folder
        #classes = np.load(os.path.join(dataset_folder, 'classes.npy'))


        test_loader = DataLoader(test_set, batch_size=config.eval.batch_size,  
                                    shuffle=False, num_workers=config.eval.num_workers)

        self.loaders = {'test': test_loader}

        # create dictionaries for character to index and index to character 
        # 0 index is reserved for CTC blank
        cdict = {c:(i+1) for i,c in enumerate(classes)}
        icdict = {(i+1):c for i,c in enumerate(classes)}

        self.classes = {
            'classes': classes,
            'c2i': cdict,
            'i2c': icdict
        }

    def prepare_net(self):

        config = self.config

        device = config.device

        print('Preparing Net - Architectural elements:')
        print(config.arch)

        classes = self.classes['classes']

        net = HTRNet(config.arch, len(classes) + 1)
        
        if config.resume is not None:
            print('resuming from checkpoint: {}'.format(config.resume))
            load_dict = torch.load(config.resume)
            load_status = net.load_state_dict(load_dict, strict=True)
            print(load_status)
        net.to(device)

        # print number of parameters
        n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print('Number of parameters: {}'.format(n_params))

        self.net = net

    def decode(self, tdec, tdict, blank_id=0):
        
        tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
        dec_transcr = ''.join([tdict[t] for t in tt if t != blank_id])
        
        return dec_transcr
    
    def test(self, epoch, tset='test'):

        config = self.config
        device = config.device

        self.net.eval()

        if tset=='test':
            loader = self.loaders['test']
        elif tset=='val':
            loader = self.loaders['val']
        else:
            print("not recognized set in test function")

        print('####################### Evaluating {} set at epoch {} #######################'.format(tset, epoch))
        
        cer, wer = CER(), WER(mode=config.eval.wer_mode)
        for (imgs, transcrs) in tqdm.tqdm(loader):

            imgs = imgs.to(device)
            with torch.no_grad():
                o = self.net(imgs)
            # if o tuple keep only the first element
            if config.arch.head_type == 'both':
                o = o[0]
            
            tdecs = o.argmax(2).permute(1, 0).cpu().numpy().squeeze()

            for tdec, transcr in zip(tdecs, transcrs):
                transcr = transcr.strip()
                dec_transcr = self.decode(tdec, self.classes['i2c']).strip()

                cer.update(dec_transcr, transcr)
                wer.update(dec_transcr, transcr)
        
        cer_score = cer.score()
        wer_score = wer.score()

        print('CER at epoch {}: {:.4f}'.format(epoch, cer_score))
        print('WER at epoch {}: {:.4f}'.format(epoch, wer_score))

        self.net.train()
        return (cer_score, wer_score)


def parse_args(file_name):
    conf = OmegaConf.load(file_name)

    OmegaConf.set_struct(conf, True)

    #sys.argv = [sys.argv[0]] + sys.argv[2:] # Remove the configuration file name from sys.argv

    conf.merge_with_cli()
    return conf

"""
if __name__ == '__main__':
    # ----------------------- initialize configuration ----------------------- #
    config = parse_args()
    max_epochs = config.train.num_epochs

    htr_eval = HTREval(config)

    htr_eval.test(0, 'val')
    htr_eval.test(0, 'test')

"""
def natural_key(text):
    return [int(chunk) if chunk.isdigit() else chunk.lower() 
            for chunk in re.split('(\d+)', text)]

def test_all_models(vs_mdl, dataset_name):
    print(f"{saved_directory}/{vs_mdl}")
    config.resume = f"{saved_directory}/{vs_mdl}"
    htr_eval = HTREval(config)
    res_eval = htr_eval.test(0, dataset_name)
    return res_eval

#Read and store model names from the saved direcoty
#results_cer_df = pd.read_csv("results_old_1/results_cer.csv")
#results_wer_df = pd.read_csv("results_old_1/results_wer.csv")


#saved_directory = config.model_save_dir

#p = saved_directory.strip().split('_')
#dir = f"{p[-2]}_{p[-1]}"

print (dir)

# saved_models = []
# for filename in os.listdir(saved_directory):
#     if os.path.isfile(os.path.join(saved_directory, filename)):
#         if (filename != f'results_test_{dir}.pkl') and (filename != f'results_val_{dir}.pkl') and (filename != 'logfile.txt'):
#             saved_models.append(filename)
# saved_models = sorted(saved_models, key=natural_key)

saved_models = "htrnet_best_cer.pt"



print(saved_models)

# results_test = test_all_models(saved_models, 'test')
# #results_val = test_all_models(saved_models, 'val')

# with open(f'{saved_directory}/results_test_{dir}.pkl', 'wb') as file:
#     pickle.dump(results_test, file)
# with open(f'{saved_directory}/results_val_{dir}.pkl', 'wb') as file:
#     pickle.dump(results_val, file)



df_splits = ["CLEAN","MIXED","MIXED_2","MIXED_3","SINGLE_LINE","DOUBLE_LINE","DIAGONAL","CROSS","WAVE","ZIG_ZAG","SCRATCH"] #
#df_splits = ["SCRATCH"] 

# experiment_trining = {
#     "CLEAN": "800_96",
#     "MIXED": "800_132",
#     "DIAGONAL": "800_134",
#     "WAVE": "800_138",
#     "ZIG_ZAG": "800_139",
#     "SCRATCH": "800_136",
#     "CROSS": "800_133",
#     "SINGLE_LINE": "800_137",
#     "DOUBLE_LINE": "800_135",
#     "Combo_WSD": "800_140"
# }

# experiment_trining = {
#     "CLEAN": "167",
#     "CLEAN": "168",
#     "CLEAN": "169",
#     "MIXED": "170",
#     "MIXED": "171",
#     "MIXED": "172",
#     "SINGLE_LINE": "102",
#     "SINGLE_LINE": "174",
#     "DOUBLE_LINE": "100",
#     "DOUBLE_LINE": "175",
#     "DIAGONAL": "99",
#     "DIAGONAL": "176",
#     "CROSS": "98",
#     "CROSS": "177",
#     "WAVE": "103",
#     "WAVE": "178",
#     "ZIG_ZAG": "104",
#     "ZIG_ZAG": "179",
#     "SCRATCH": "101",
#     "SCRATCH": "180",
# }

experiment_trining = {
    "167": "CLEAN",
    "168": "CLEAN",
    "169": "CLEAN",
    "158": "MIXED",
    "189": "MIXED",
    "199": "MIXED",
    "207_1": "SINGLE_LINE",
    "207_2": "SINGLE_LINE",
    "207_3": "SINGLE_LINE",
    "208_1": "DOUBLE_LINE",
    "208_2": "DOUBLE_LINE",
    "208_3": "DOUBLE_LINE",
    "209_1": "DIAGONAL",
    "209_2": "DIAGONAL",
    "209_3": "DIAGONAL",
    "210_1": "CROSS",
    "210_2": "CROSS",
    "210_3": "CROSS",
    "211_1": "WAVE",
    "211_2": "WAVE",
    "211_3": "WAVE",
    "212_1": "ZIG_ZAG",
    "212_2": "ZIG_ZAG",
    "212_3": "ZIG_ZAG",
    "213_1": "SCRATCH",
    "213_2": "SCRATCH",
    "213_3": "SCRATCH"
}

# experiment_trining = {
#     "200": "SINGLE_LINE",
#     "201": "DOUBLE_LINE",
#     "202": "DIAGONAL",
#     "203": "CROSS",
#     "204": "WAVE",
#     "205": "ZIG_ZAG",
#     "206": "SCRATCH",
# }


main_cer = []
main_wer = []

config = parse_args(f"configs/config_800_207.yml")
saved_directory = f"../../HTR-best-practices-strike2/results/saved_models_800_207_1"

config_probability2_clean = 0.0
config_probability2_striked_types = {"SINGLE_LINE": 1.0}

results_test = test_all_models(saved_models, 'test')

print(results_test)

# #df = pd.DataFrame(columns= ["Trained_on", "Score", "Model_ID"] + df_splits)
# df = pd.read_csv('results_types_all_cross.csv')

# for case in experiment_trining:
#     saved_directory = f"results/saved_models_800_{case}"
#     print(f"Trained Model: {case}")

#     row_cer = {"Trained_on": experiment_trining[case], "Score": "CER", "Model_ID": case}
#     row_wer = {"Trained_on": experiment_trining[case], "Score": "WER", "Model_ID": case}
#     p=0 #CER: p=0 | WER: p=1

#     for idx, split in enumerate(df_splits):
#         test_dataset = split
#         if split == "CLEAN":
#             config_probability2_clean = 0.0
#             config_probability2_striked_types = {"CLEAN": 1.0}
        
#         if split == "MIXED": #This need to be run on te fixed Mixed set by dataloader2.py which is the "evaluate_table_strike_removed.py"
#             config_probability2_clean = 0.0
#             config_probability2_striked_types = {"MIXED": 1.0}

#         if split == "MIXED_2": 
#             config_probability2_clean = 0.0
#             config_probability2_striked_types = {"MIXED_2": 1.0}        
            
#         if split == "MIXED_3": 
#             config_probability2_clean = 0.0
#             config_probability2_striked_types = {"MIXED_3": 1.0}

#         if split == "SINGLE_LINE":
#             config_probability2_clean = 0.0
#             config_probability2_striked_types = {"SINGLE_LINE": 1.0}
        
#         if split == "DOUBLE_LINE":
#             config_probability2_clean = 0.0
#             config_probability2_striked_types = {"DOUBLE_LINE": 1.0}        
        
#         if split == "DIAGONAL":
#             config_probability2_clean = 0.0
#             config_probability2_striked_types = {"DIAGONAL": 1.0}

#         if split == "CROSS":
#             config_probability2_clean = 0.0
#             config_probability2_striked_types = {"CROSS": 1.0}
        
#         if split == "WAVE":
#             config_probability2_clean = 0.0
#             config_probability2_striked_types = {"WAVE": 1.0}
        
#         if split == "ZIG_ZAG":
#             config_probability2_clean = 0.0
#             config_probability2_striked_types = {"ZIG_ZAG": 1.0}
        
#         if split == "SCRATCH":
#             config_probability2_clean = 0.0
#             config_probability2_striked_types = {"SCRATCH": 1.0}    
        



#         # config = parse_args(f"configs/config_800_{case}.yml")
#         config = parse_args(f"configs/config_800_207.yml")

#         print(split)
#         print(config.probability.clean)
#         print(config.probability.striked_types)

#         results_test = test_all_models(saved_models, 'test')
#         print(results_test)

#         row_cer[split] = results_test[0]
#         row_wer[split] = results_test[1]

#     df.loc[len(df)] = row_cer
#     df.loc[len(df)] = row_wer


# df.to_csv('results_types_all_cross.csv', index=False)
