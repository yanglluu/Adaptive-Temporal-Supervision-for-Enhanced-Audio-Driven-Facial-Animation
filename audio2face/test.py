import os
import numpy as np
import torch
import argparse
import scipy
import copy
from scipy.io import wavfile
import pickle
from    model import TfaceGAN
import glob
from os.path import join, exists, abspath, dirname
from modules.dcaw_controller import DCAWController

parser = argparse.ArgumentParser(description='Test_setting')
parser.add_argument('--audiopath', type=str, default='../examples/audio_preprocessed/test1.pkl')
parser.add_argument('--checkpath', type=str, default='./checkpoint/train1/Gen-10.mdl')
parser.add_argument('--outpath', type=str, default = '../examples/test-result')
parser.add_argument('--use_dcaw', action='store_true', help='Enable adaptive windowing during inference')
parser.add_argument('--soft_gate', action='store_true', help='Enable differentiable soft gating for attention windowing (inference)')
parser.add_argument('--soft_wmin', type=int, default=64)
parser.add_argument('--soft_wmax', type=int, default=256)

opt = parser.parse_args()

num_params = 71
out_path = opt.outpath

if not os.path.exists(out_path):
	os.makedirs(out_path)

audio_list = glob.glob(opt.audiopath)

for audio_path in audio_list:
    print (audio_path)
    
    processed_audio = pickle.load(open(audio_path, 'rb'), encoding=' iso-8859-1')
    

    modelgen = TfaceGAN().cuda()

    modelgen.load_state_dict(torch.load(opt.checkpath))
    modelgen.eval()

    processed_audio = torch.Tensor(processed_audio)
    audioname = audio_path.split('/')[-1].replace('.pkl', '')

    faceparams = np.zeros((processed_audio.shape[0], num_params), float)

    frames_out_path = os.path.join(out_path, audioname+'.npz')
    firstpose = torch.zeros([1,num_params],dtype=torch.float32).unsqueeze(0)
    

    with torch.no_grad():
        if opt.use_dcaw:
            controller = DCAWController()
            # Decide stride from first 256 frames
            est_audio = processed_audio[:min(256, processed_audio.shape[0])]
            stride = controller.suggest_stride(est_audio)
        else:
            stride = 127

        i = 0
        while i <= processed_audio.shape[0]-128:
            audio = processed_audio[i:i+128,:,:].unsqueeze(0).cuda()
            if opt.use_dcaw:
                # build a central mask based on controller decision for this window (64/128/256)
                W_t, _ = controller.compute(audio.squeeze(0).cpu().numpy(), None)
                W_sel = int(np.median(W_t))
                mask_np = controller.make_window_mask(128, W_sel)
                attn_mask = torch.from_numpy(mask_np.astype('float32')).unsqueeze(0).cuda()
                # no direct use in inference loss; we adapt stride via controller
                if W_sel <= 64:
                    stride = 64
                elif W_sel <= 128:
                    stride = 96
                else:
                    stride = 127

            if opt.use_dcaw:
                _faceparam = modelgen(audio,firstpose.cuda(), attn_mask=attn_mask, soft_gate=opt.soft_gate, soft_wmin=opt.soft_wmin, soft_wmax=opt.soft_wmax)
            else:
                _faceparam = modelgen(audio,firstpose.cuda())
            firstpose = _faceparam[:,127:128,:]
            faceparams[i:i+128,:] = _faceparam[0,:,:].cpu().numpy()
            i += stride

        # Ensure last window is covered
        j = max(0, processed_audio.shape[0]-128)
        audio = processed_audio[j:j+128,:,:].unsqueeze(0).cuda()
        if opt.use_dcaw:
            # build a central mask for the last window
            W_t, _ = controller.compute(audio.squeeze(0).cpu().numpy(), None)
            W_sel = int(np.median(W_t))
            mask_np = controller.make_window_mask(128, W_sel)
            attn_mask = torch.from_numpy(mask_np.astype('float32')).unsqueeze(0).cuda()
            _faceparam = modelgen(audio,firstpose.cuda(), attn_mask=attn_mask, soft_gate=opt.soft_gate, soft_wmin=opt.soft_wmin, soft_wmax=opt.soft_wmax)
        else:
            _faceparam = modelgen(audio,firstpose.cuda())
        faceparams[j:j+128,:] = _faceparam[0,:,:].cpu().numpy()

        np.savez(frames_out_path, face = faceparams)


