import binvox_rw
import numpy as np
import plotly.graph_objects as go
from models.encoder import Encoder
from models.decoder import Decoder
from models.merger import Merger
from models.refiner import Refiner
from config import cfg
import torch
from datetime import datetime as dt
import utils.data_transforms
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg.CONST.WEIGHTS='saved_models/Pix2Vox-A-ShapeNet.pth'


def read_binvox(file) -> np.ndarray:
    model = binvox_rw.read_as_3d_array(file)
    return model.data.astype(np.uint8)


def voxel_to_plotly(voxels):
    x, y, z = voxels.nonzero()
    fig = go.Figure(data=[
        go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=3, color=z, colorscale='Viridis', opacity=0.7)
        )
    ])
    fig.update_layout(scene=dict(aspectmode='data'))
    return fig



IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
test_transforms = utils.data_transforms.Compose([
    utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
    utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
    utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
    utils.data_transforms.ToTensor(),
])


# def save_binvox(volume):

#     if volume.ndim == 4:
#         volume = volume.squeeze()

#     volume = (volume >= 0.5).astype(np.uint8)

#     dims = volume.shape

#     print("the dimension after above operation:", dims)

#     translate = [0.0, 0.0, 0.0]       # position offset
#     scale = 1.0                       # scaling factor
#     axis_order = 'xyz'               # or 'xzy', etc., depending on your app

#     filename = f"output.binvox"
#     save_path=os.path.join(save_path,filename)

#     with open(filename, 'wb') as f:
#         model = binvox_rw.Voxels(volume, dims, translate, scale, axis_order)
#         model.write(f)



def predict_voxel_from_images(rendering_images):
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

        # print(rendering_images.shape)
        image_features = encoder(transformed_images)
        print(image_features.shape)
        raw_features, generated_volume = decoder(image_features)
        print(generated_volume.shape)


        if cfg.NETWORK.USE_MERGER:
            generated_volume = merger(raw_features, generated_volume)
        else:
            generated_volume = torch.mean(generated_volume, dim=1)


        # encoder_loss = bce_loss(generated_volume, ground_truth_volume) * 10


        if cfg.NETWORK.USE_REFINER:
                generated_volume = refiner(generated_volume)
                # refiner_loss = bce_loss(generated_volume, ground_truth_volume) * 10
        else:
                # refiner_loss = encoder_loss
                pass


        generated_volume=generated_volume.squeeze(0)
        gv = generated_volume.cpu().numpy()

        gv = (gv >= 0.5).astype(np.uint8)




    torch.cuda.empty_cache()
    return gv


        # for saving

        # get_volume_views(gv,save_dir,img_index)

        # print("After passing to get_volume views: ")
        # count_0 = sum(1 for plane in gv for row in plane for val in row if val < 0.5)
        # print(count_0)

        # count_1=sum(1 for plane in gv for row in plane for val in row if val >= 0.5)
        # print(count_1)

        # save_binvox(gv,save_dir,img_index)

        # with open(gt_binvox_path, 'rb') as f:
        #     gtv = utils.binvox_rw.read_as_3d_array(f)
        #     gtv = gtv.data.astype(np.float32)

        # print("For the ground truth: ")
        # save_dir_gt='model_inference_single_view/ground_truth'
        # get_volume_views(gtv,save_dir_gt,img_index)
        # save_binvox(gtv,save_dir_gt,img_index)
