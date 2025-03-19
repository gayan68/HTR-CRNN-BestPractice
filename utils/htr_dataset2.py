import io,os
import numpy as np
import torch
from torch.utils.data import Dataset
from os.path import isfile
from skimage.transform import resize
from utils.preprocessing import load_image, preprocess
import pandas as pd
import random

class HTRDatasetBoth(Dataset): 
    def __init__(self, 
        config,
        subset: str = 'train',                          #Name of dataset subset to be loaded. (e.g. ''train', 'val', 'test')
        fixed_size: tuple =(64, None), #128          #Resize inputs to this size
        transforms: list = None,                      #List of augmentation transform functions to be applied on each input
        character_classes: list = None,               #If 'None', these will be autocomputed. Otherwise, a list of characters is expected.
        ):
        self.config = config
        self.basefolder = config.data.path
        self.dataset = config.data.dataset
        self.subset = subset
        self.fixed_size = fixed_size
        self.transforms = transforms
        self.character_classes = character_classes


        # load gt.txt from basefolder - each line contains a path to an image and its transcription
        data = []
        #with open(os.path.join(self.basefolder_clean, subset, 'gt.txt'), 'r') as f:
        with open(os.path.join(self.basefolder, subset, 'images', self.dataset, 'gt.txt'), 'r') as f:
            for line in f:
                img_id, transcr = line.strip().split(' ')[0], ' '.join(line.strip().split(' ')[1:])

                ######### GAYAN #############
                transcr = transcr.replace(" ", "")
                # "We 'll" -> "We'll"
                special_cases  = ["s", "d", "ll", "m", "ve", "t", "re"]
                # lower-case 
                for cc in special_cases:
                    transcr = transcr.replace("|\'" + cc, "\'" + cc)
                    transcr = transcr.replace("|\'" + cc.upper(), "\'" + cc.upper())
                transcr = transcr.replace("|", " ")
                ######### GAYAN End #########
                if (config.preprocess.ignore_1_chr == False) or (len(transcr)>1):
                    if (len(config.preprocess.ignore_chars) == 0) or (transcr not in config.preprocess.ignore_chars):
                        img_path = os.path.join(self.basefolder, subset, "images", self.dataset , img_id + '.png')
                        data += [(img_path, transcr)]

        for i in range(10):
            print(data[i])


        self.data = data

        if self.character_classes is None:
            res = set()
            for _,transcr in data:
                res.update(list(transcr))
            res = sorted(list(res))
            print('Character classes: {} ({} different characters)'.format(res, len(res)))
            self.character_classes = res 
            

    def __getitem__(self, index):
        img_path = self.data[index][0]
        transcr = " " + self.data[index][1] + " "
        fheight, fwidth = self.fixed_size[0], self.fixed_size[1]

        img = load_image(img_path)

        if self.subset == 'train':
            nwidth = int(np.random.uniform(.75, 1.25) * img.shape[1])
            nheight = int((np.random.uniform(.9, 1.1) * img.shape[0] / img.shape[1]) * nwidth)
            nwidth = img.shape[1] if nwidth == 0 else nwidth
            nheight = img.shape[0] if nheight == 0 else nheight
            img = resize(image=img, output_shape=(nheight, nwidth)).astype(np.float32)

        img = preprocess(self.config, img, (fheight, fwidth))

        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        img = torch.Tensor(img).float().unsqueeze(0)
        return img, transcr
    
    def __len__(self):
        return len(self.data)
