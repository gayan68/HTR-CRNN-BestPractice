import argparse
from omegaconf import OmegaConf

import sys
import os
import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.htr_dataset3 import HTRDatasetBoth

from models import HTRNet
from utils.transforms import aug_transforms

import torch.nn.functional as F
import wandb

from utils.metrics import CER, WER

from skimage import img_as_ubyte
import skimage.io as img_io
import time

class HTRTrainer(nn.Module):
    def __init__(self, config):
        super(HTRTrainer, self).__init__()
        self.config = config

        self.prepare_dataloaders()
        self.prepare_net()
        self.prepare_losses()
        self.prepare_optimizers()

    def prepare_dataloaders(self):

        config = self.config

        # prepare datset loader
        dataset_folder = config.data.clean_path
        #dataset_folder_strike = config.data.strike_path
        fixed_size = (config.preproc.image_height, config.preproc.image_width)


        t1 = time.time()
        train_set = HTRDatasetBoth(config, 'train', fixed_size=fixed_size, transforms=aug_transforms(config.train.aug_prob))
        print(f"EXE TIME: {time.time()-t1}")
        
        classes = train_set.character_classes
        print('# training lines ' + str(train_set.__len__()))
        #img_io.imsave("image1.png", img_as_ubyte(train_set[1][0]))

        val_set = HTRDatasetBoth(config, 'val', fixed_size=fixed_size, transforms=None)
        print('# validation lines ' + str(val_set.__len__()))

        test_set = HTRDatasetBoth(config, 'test', fixed_size=fixed_size, transforms=None)
        print('# testing lines ' + str(test_set.__len__()))
        
        # augmentation using data sampler
        train_loader = DataLoader(train_set, batch_size=config.train.batch_size, 
                                  shuffle=True, num_workers=config.train.num_workers)
        if val_set is not None:
            val_loader = DataLoader(val_set, batch_size=config.eval.batch_size,  
                                    shuffle=False, num_workers=config.eval.num_workers)
        test_loader = DataLoader(test_set, batch_size=config.eval.batch_size,  
                                    shuffle=False, num_workers=config.eval.num_workers)

        self.loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

        # add space to classes, if not already there
        classes += ' ' 
        classes = np.unique(classes)

        # save classes in data folder
        np.save(os.path.join(dataset_folder, 'classes.npy'), classes)

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
        
        #print(net.state_dict().keys())
        #print("########################################################")
        if config.resume is not None:
            print('resuming from checkpoint: {}'.format(config.resume))
            load_dict = torch.load(config.resume)
            
            #### Added by Gayan - Different Achi - Pretrain ###
            """
            keys_to_remove = ["top.cnn.1.weight", "top.cnn.1.bias"]
            for key in keys_to_remove:
                if key in load_dict:  # Check to avoid KeyError
                    del load_dict[key]
            """
            ####################################################
            #print(load_disct.keys())

            load_status = net.load_state_dict(load_dict, strict=True)
            print(load_status)
        net.to(device)

        # print number of parameters
        n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print('Number of parameters: {}'.format(n_params))

        self.net = net

    def prepare_losses(self):
        self.ctc_loss = lambda y, t, ly, lt: nn.CTCLoss(reduction='sum', zero_infinity=True)(F.log_softmax(y, dim=2), t, ly, lt) /self.config.train.batch_size

    def prepare_optimizers(self):
        config = self.config
        optimizer = torch.optim.AdamW(self.net.parameters(), config.train.lr, weight_decay=0.00005)

        self.optimizer = optimizer

        max_epochs = config.train.num_epochs
        if config.train.scheduler == 'mstep':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(.5*max_epochs), int(.75*max_epochs)])
        else:
            raise NotImplementedError('Alternative schedulers not implemented yet')

    def decode(self, tdec, tdict, blank_id=0):
        
        tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
        dec_transcr = ''.join([tdict[t] for t in tt if t != blank_id])
        
        return dec_transcr
                
    def sample_decoding(self):

        # get a random image from the test set
        img, transcr = self.loaders['val'].dataset[np.random.randint(0, len(self.loaders['val'].dataset))]

        img = img.unsqueeze(0).to(self.config.device)

        self.net.eval()
        with torch.no_grad():
            tst_o = self.net(img)
            if self.config.arch.head_type == 'both':
                tst_o = tst_o[0]

        self.net.train()

        tdec = tst_o.argmax(2).permute(1, 0).cpu().numpy().squeeze()
        # remove duplicates
        dec_transcr = self.decode(tdec, self.classes['i2c'])

        print('orig:: ' + transcr.strip())
        print('pred:: ' + dec_transcr.strip())


    def train(self, epoch):

        config = self.config
        device = config.device

        self.net.train()

        epoch_loss = 0.0

        t = tqdm.tqdm(self.loaders['train'])
        num_batches = len(self.loaders['train'])
        t.set_description('Epoch {}'.format(epoch))
        for iter_idx, (img, transcr) in enumerate(t):
            self.optimizer.zero_grad()

            img = img.to(device)
            if(img.max().item() > 1) or (img.min().item()<0):
                #print("##### ERROR######")
                #print(f"Before: Max value: {img.max().item()} | Min value: {img.min().item()}")
                img = img.clamp(0,1)  #Added ITR: 31
            #print(f"After: Max value: {img.max().item()} | Min value: {img.min().item()}")

            if config.arch.head_type == "both":
                output, aux_output = self.net(img)
            else:
                output = self.net(img)

            act_lens = torch.IntTensor(img.size(0)*[output.size(0)])
            labels = torch.IntTensor([self.classes['c2i'][c] for c in ''.join(transcr)])
            label_lens = torch.IntTensor([len(t) for t in transcr])

            loss_val = self.ctc_loss(output, labels, act_lens, label_lens) #Added ITR: 31   
            tloss_val = loss_val.item()    
            epoch_loss += tloss_val

            if config.arch.head_type == "both":
                loss_val += 0.1 * self.ctc_loss(aux_output.cpu(), labels, act_lens, label_lens) #Added ITR: 31

            
            current_lr = self.optimizer.param_groups[0]['lr']

            loss_val.backward()
            self.optimizer.step()    

            t.set_postfix(values='train_loss : {:.2f}'.format(tloss_val))

        average_loss = epoch_loss / num_batches

        self.sample_decoding()

        return average_loss, current_lr
            
            
    def validate(self, epoch):

        config = self.config
        device = config.device

        self.net.eval()

        epoch_loss = 0.0

        t = tqdm.tqdm(self.loaders['val'])
        num_batches = len(self.loaders['val'])
        for iter_idx, (img, transcr) in enumerate(t):
            img = img.to(device)

            if config.arch.head_type == "both":
                output, aux_output = self.net(img)
            else:
                output = self.net(img)

            act_lens = torch.IntTensor(img.size(0)*[output.size(0)])
            labels = torch.IntTensor([self.classes['c2i'][c] for c in ''.join(transcr)])
            label_lens = torch.IntTensor([len(t) for t in transcr])

            loss_val = self.ctc_loss(output.cpu(), labels, act_lens, label_lens)
            # GAYAN: I don't thin we need CTC shortcut loss for the validation
            """
            if config.arch.head_type == "both":
                loss_val += 0.1 * self.ctc_loss(aux_output.cpu(), labels, act_lens, label_lens)
            """
            epoch_loss += loss_val.item()

            t.set_postfix(values='val_loss : {:.2f}'.format(loss_val.item()))

        average_loss = epoch_loss / num_batches
        #wandb.log({"epoch_val_loss": average_loss})

        return average_loss

    
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

        #wandb.log({"cer": cer_score, "wer": wer_score})

        print('CER at epoch {}: {:.3f}'.format(epoch, cer_score))
        print('WER at epoch {}: {:.3f}'.format(epoch, wer_score))

        self.net.train()

        return cer_score, wer_score
        

    def save(self, epoch):
        print('####################### Saving model at epoch {} #######################'.format(epoch))
        if not os.path.exists(self.config.model_save_dir):
            os.makedirs(self.config.model_save_dir)

        torch.save(self.net.cpu().state_dict(), self.config.model_save_dir + '/htrnet_{}.pt'.format(epoch))
    
        self.net.to(self.config.device)


def parse_args():
    conf = OmegaConf.load(sys.argv[1])

    OmegaConf.set_struct(conf, True)

    sys.argv = [sys.argv[0]] + sys.argv[2:] # Remove the configuration file name from sys.argv

    conf.merge_with_cli()
    return conf

def mylog(config, log_message):
    with open(f"{config.model_save_dir}/logfile.txt", "a") as log_file:
        log_file.write(log_message + "\n")

if __name__ == '__main__':
    # ----------------------- initialize configuration ----------------------- #
    config = parse_args()
    max_epochs = config.train.num_epochs


    wandb_log = True  #Set wandb web log

    # ----------------------- initialize wandb ------------------------------- #
    if wandb_log:
        wandb.init(
            # set the wandb project where this run will be logged
            project="HTR_BestPractice",

            # track hyperparameters and run metadata
            config={
            "learning_rate": config.train.lr,
            "architecture": "CNN+LSTM",
            "dataset": "IAM",
            "epochs": config.train.num_epochs,
            }
        )
        print("run name WANDB : " + str(wandb.run.name))
    
    best_val_loss = 10000
    best_cer_loss = 10000

    htr_trainer = HTRTrainer(config)

    cnt = 1
    print('Training Started!')
    cer_score, wer_score = htr_trainer.test(0, 'val')
    for epoch in range(1, max_epochs + 1):
        train_loss, current_lr = htr_trainer.train(epoch)
        val_loss = htr_trainer.validate(epoch)
        cer_score, wer_score = htr_trainer.test(epoch, 'val')
        htr_trainer.scheduler.step()
        
        # save and evaluate the current model
        if epoch % config.train.save_every_k_epochs == 0:
            htr_trainer.save(epoch)
            mylog(config, f"Mode: every_{config.train.save_every_k_epochs}_epochs | Epoch: {epoch} | CER: {cer_score} | WER: {wer_score}")

        #save best model based on Validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            htr_trainer.save("best")
            mylog(config, f"Mode: best_validation | Epoch: {epoch} | CER: {cer_score} | WER: {wer_score}")
        
        if cer_score < best_cer_loss:
            best_cer_loss = cer_score
            htr_trainer.save("best_cer")
            mylog(config, f"Mode: best_cer | Epoch: {epoch} | CER: {cer_score} | WER: {wer_score}")

        #Upload logs to Wandb
        if wandb_log:
            wandb.log({"Train_loss": train_loss, "Valid_loss": val_loss, "current_lr": current_lr, "CER": cer_score, "WER": wer_score})

    # save the final model
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    torch.save(htr_trainer.net.cpu().state_dict(), config.model_save_dir + '/{}'.format(config.save))
