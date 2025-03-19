import logging
import sys
import os
import argparse
from omegaconf import OmegaConf


basefolder = 'data/IAM/processed_words/'
subset = 'test'

# load gt.txt from basefolder - each line contains a path to an image and its transcription
count = 0
with open(os.path.join(basefolder, subset, 'gt.txt'), 'r') as f:
    for line in f:
        img_path, transcr = line.strip().split(' ')[0], ' '.join(line.strip().split(' ')[1:])
        if (len(transcr)==1):
            count += 1

print(count)

