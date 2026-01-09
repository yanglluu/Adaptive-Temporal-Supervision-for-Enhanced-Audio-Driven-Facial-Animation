import os
import glob
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from modules.dcaw_controller import DCAWController

class Facial_Dataset(Dataset):
    def __init__(self, audio_paths,npz_paths,cvs_paths, use_dcaw: bool = True):
        self.audio_path_list = audio_paths
        self.mesh_param_path_list = npz_paths
        self.blink_path_list = cvs_paths 
        self.frames = 128
        self.use_dcaw = use_dcaw
        self.controller = DCAWController() if self.use_dcaw else None

        self.dataset_audio = []
        self.dataset_exp_param = []
        self.dataset_idx = []
        base =0

        for audio_path, param_path, blink_path in zip(self.audio_path_list, self.mesh_param_path_list, self.blink_path_list):
            audio_name = audio_path.split('/')[-1].replace('.pkl', '')
            param_name = param_path.split('/')[-1].replace('.npz', '')
            try:
                assert audio_name == param_name
            except:
                print (audio_name, param_name)
                
            audio = pickle.load(open(audio_path, 'rb'), encoding=' iso-8859-1')
            params = np.load(open(param_path, 'rb'))
            params = params['face']

            blinkinfo=pd.read_csv(blink_path)
            aublink = blinkinfo['AU45_r'].values

            std1 = np.std(params, axis=0)
            mean1 = np.mean(params,axis=0)

            for i in range(6):
                params[:,i] = (params[:,i]-mean1[i])/std1[i]
            
            
            min_num_frame = min(params.shape[0], audio.shape[0])

            if params.shape[0] != audio.shape[0]:
                params=params[0:min_num_frame,:]
                audio=audio[0:min_num_frame,:,:]
                aublink = aublink[0:min_num_frame]

            aublink = aublink[:, np.newaxis]

            self.dataset_audio.append(audio)
            self.dataset_exp_param.append(np.concatenate(( params[:,:6], aublink, params[:,7:71]),axis=1))
            # keep dense index list; dynamic slicing will be applied in __getitem__
            self.dataset_idx += list(np.arange(0, min_num_frame-self.frames, 1) + base)
            base+= min_num_frame


        self.dataset_audio = torch.Tensor(np.concatenate(self.dataset_audio))
        self.dataset_exp_param = torch.Tensor(np.concatenate(self.dataset_exp_param))
        self.dataset_idx = torch.Tensor(np.array(self.dataset_idx)).to(torch.long)
        print(max(self.dataset_idx))
        assert self.dataset_audio.shape[0] == self.dataset_exp_param.shape[0] 


    def __len__(self):
        return len(self.dataset_idx)

    def __getitem__(self, idx):
        start = int(self.dataset_idx[idx])
        end = start + self.frames
        audio_win = self.dataset_audio[start:end]
        param_win = self.dataset_exp_param[start:end]

        if self.use_dcaw and self.controller is not None:
            # Compute S_t and window selection mask for current window
            # audio_win: [T, W, C] (DeepSpeech windows) or [T, C], param_win: [T, P]
            W_t, S_t = self.controller.compute(audio_win, param_win)
            # Use median-selected window size to build a centered mask
            W_sel = int(np.median(W_t))
            mask_np = self.controller.make_window_mask(self.frames, W_sel)
            s_weights = torch.from_numpy(S_t.astype('float32'))
            win_mask = torch.from_numpy(mask_np.astype('float32'))
        else:
            s_weights = torch.ones((self.frames,), dtype=torch.float32)
            win_mask = torch.ones((self.frames,), dtype=torch.float32)

        return audio_win, param_win, s_weights, win_mask
