import cv2
import os
import numpy as np
from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger

import torch
from config import cfg
from datetime import datetime as dt
import utils.data_transforms
import utils.binvox_rw



device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def save_binvox(volume, save_path,index):

    if volume.ndim == 4:
        volume = volume.squeeze()

    volume = (volume >= 0.5).astype(np.uint8)

    dims = volume.shape

    count_0 = sum(1 for plane in volume for row in plane for val in row if val == 0)
    print("The number of 0s: ",count_0)

    count_1=sum(1 for plane in volume for row in plane for val in row if val == 1)
    print("The number of 1: ",count_1)

    print("the dimension after above operation:", dims)

    translate = [0.0, 0.0, 0.0]       # position offset
    scale = 1.0                       # scaling factor
    axis_order = 'xyz'               # or 'xzy', etc., depending on your app

    filename = f"{index}.binvox"
    save_path=os.path.join(save_path,filename)

    with open(save_path, 'wb') as f:
        model = utils.binvox_rw.Voxels(volume, dims, translate, scale, axis_order)
        model.write(f)


IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
test_transforms = utils.data_transforms.Compose([
    utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
    utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
    utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
    utils.data_transforms.ToTensor(),
])



model_dir='test_images/test_02'
img_paths=[]

img_index=0
n_views=5

for i in range(n_views):
    img_path = os.path.join(model_dir, f"{i:02}.png")
    img_paths.append(img_path)

print(img_paths)

rendering_images=[]
for img_path in img_paths:
    rendering_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
    # print(rendering_image.shape)
    rendering_images.append(rendering_image)


transformed_images = test_transforms(rendering_images)

encoder = Encoder(cfg)
decoder = Decoder(cfg)
refiner = Refiner(cfg)
merger = Merger(cfg)


if torch.cuda.is_available():
        encoder = torch.nn.DataParallel(encoder).cuda()
        decoder = torch.nn.DataParallel(decoder).cuda()
        refiner = torch.nn.DataParallel(refiner).cuda()
        merger = torch.nn.DataParallel(merger).cuda()

print('[INFO] %s Loading weights from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
checkpoint = torch.load(cfg.CONST.WEIGHTS)

epoch_idx = checkpoint['epoch_idx']
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])

if cfg.NETWORK.USE_REFINER:
        refiner.load_state_dict(checkpoint['refiner_state_dict'])
if cfg.NETWORK.USE_MERGER:
        merger.load_state_dict(checkpoint['merger_state_dict'])


encoder.eval()
decoder.eval()
merger.eval()
refiner.eval()


with torch.no_grad():

    transformed_images = transformed_images.unsqueeze(0) #adding the batch_dim
    transformed_images = transformed_images.to(device)

    print(rendering_image.shape)
    image_features = encoder(transformed_images)
    print(image_features.shape)
    raw_features, generated_volume = decoder(image_features)
    print(generated_volume.shape)


    if cfg.NETWORK.USE_MERGER:
        generated_volume = merger(raw_features, generated_volume)
    else:
        generated_volume = torch.mean(generated_volume, dim=1)




    if cfg.NETWORK.USE_REFINER:
            generated_volume = refiner(generated_volume)

    else:
        pass



    # gt_dir='model_inference/gt'
    save_dir=model_dir

    generated_volume=generated_volume.squeeze(0)
    gv = generated_volume.cpu().numpy()


    save_binvox(gv,save_dir,img_index)
