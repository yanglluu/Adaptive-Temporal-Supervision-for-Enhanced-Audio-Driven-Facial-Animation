import os
from collections import OrderedDict
from options.test_options import TestOptions
from data.custom_dataset_data_loader import CreateDataset
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
import torch
from imageio import get_writer
import numpy as np
from tqdm import tqdm
import cv2
from cv2 import VideoWriter,VideoWriter_fourcc,imread,resize
import os
import glob
from os.path import join, exists, abspath, dirname
import ffmpeg
import mediapipe as mp


opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#test images = %d' % dataset_size)
model = create_model(opt).cuda()
if opt.verbose:
    print(model)

from skimage.io import imsave

img_root = '../examples/test_image/'+opt.test_id_name
if not os.path.exists(img_root):
    os.makedirs(img_root)

def fix_eye_region(image):
    mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
    results = mp_face.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        # 只处理第一张人脸
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = image.shape
        # 眼睛关键点索引（mediapipe官方定义）
        left_eye_idx = [33, 133, 160, 159, 158, 157, 173, 246]
        right_eye_idx = [362, 263, 387, 386, 385, 384, 398, 466]
        # 获取左眼、右眼坐标
        left_eye_pts = [(int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(face_landmarks.landmark) if i in left_eye_idx]
        right_eye_pts = [(int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(face_landmarks.landmark) if i in right_eye_idx]
        # 用椭圆模糊修复眼睛区域
        for eye_pts in [left_eye_pts, right_eye_pts]:
            if len(eye_pts) > 0:
                center = np.mean(eye_pts, axis=0).astype(int)
                axes = (max(1, int(0.6 * np.ptp([p[0] for p in eye_pts]))), max(1, int(0.6 * np.ptp([p[1] for p in eye_pts]))))
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.ellipse(mask, tuple(center), axes, 0, 0, 360, 255, -1)
                image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    return image

# === 设定转头阈值 ===
YAW_THRESHOLD = 20  # 仅对yaw大于20度的帧进行修复

for i, data in enumerate(dataset):
    print(i)
    label = data['label']
    cur_frame = model.inference(label)
    prev_frame = cur_frame.data[0]
    
    if i+7<= len(dataset):
        frameindex = (i+7)
    else:
        frameindex = i + 7 - len(dataset)
    # === 仅对大角度转头帧修复眼睛区域 ===
    frame_img = util.tensor2im(prev_frame)
    # 假设data中有'yaw'字段，单位为度
    yaw = data.get('yaw', 0)  # 若无yaw字段，默认0
    if abs(yaw) > YAW_THRESHOLD:
        frame_img = fix_eye_region(frame_img)
    imsave(img_root+'/{:06d}.jpg'.format(frameindex), frame_img)




fps = 30

fourcc=VideoWriter_fourcc('M','J','P','G')
videoWriter = cv2.VideoWriter(img_root+'/test_1.avi',fourcc,fps,(512,512))
im_names=os.listdir(img_root)
im_names = sorted(glob.glob(join(img_root, '*[0-9]*.jpg')))

for im_name in range(len(im_names)):
    frame=cv2.imread(im_names[im_name])


    frame = cv2.resize(frame, (512,512)) 
    print (im_name)
    videoWriter.write(frame)
print(videoWriter)
videoWriter.release()
