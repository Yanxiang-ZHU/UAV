import os
import cv2
import torch
import numpy as np

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

torch.backends.cudnn.enabled = False


_upsampler = None


def _init_upsampler(
    model_name='RealESRGAN_x4plus',
    outscale=4,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    fp32=False,
    gpu_id=None,
    denoise_strength=0.5,
):
    global _upsampler

    if _upsampler is not None:
        return _upsampler

    model_name = model_name.split('.')[0]

    # if model_name == 'RealESRGAN_x4plus':
    #     # model = RRDBNet(3, 3, 64, 23, 32, scale=4)
    #     RRDBNet(
    #         num_in_ch=3,
    #         num_out_ch=3,
    #         num_feat=64,
    #         num_block=23,
    #         num_grow_ch=32,
    #         scale=4
    #     )
    #     netscale = 4
    #     file_url = [
    #         'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    #     ]
    # elif model_name == 'RealESRNet_x4plus':
    #     # model = RRDBNet(3, 3, 64, 23, 32, scale=4)
    #     RRDBNet(
    #         num_in_ch=3,
    #         num_out_ch=3,
    #         num_feat=64,
    #         num_block=23,
    #         num_grow_ch=32,
    #         scale=4
    #     )
    #     netscale = 4
    #     file_url = [
    #         'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth'
    #     ]
    # elif model_name == 'RealESRGAN_x4plus_anime_6B':
    #     # model = RRDBNet(3, 3, 64, 6, 32, scale=4)
    #     RRDBNet(
    #         num_in_ch=3,
    #         num_out_ch=3,
    #         num_feat=64,
    #         num_block=6,
    #         num_grow_ch=32,
    #         scale=4
    #     )
    #     netscale = 4
    #     file_url = [
    #         'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'
    #     ]
    # elif model_name == 'RealESRGAN_x2plus':
    #     # model = RRDBNet(3, 3, 64, 23, 32, scale=2)
    #     RRDBNet(
    #         num_in_ch=3,
    #         num_out_ch=3,
    #         num_feat=64,
    #         num_block=23,
    #         num_grow_ch=32,
    #         scale=2
    #     )
    #     netscale = 2
    #     file_url = [
    #         'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
    #     ]
    # # elif model_name == 'realesr-animevideov3':
    # #     model = SRVGGNetCompact(3, 3, 64, 16, upscale=4, act_type='prelu')
    # #     netscale = 4
    # #     file_url = [
    # #         'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth'
    # #     ]
    # else:
    #     raise ValueError(f'Unsupported model: {model_name}')
    
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4
    )
    netscale = 4
    file_url = [
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    ]

    # model path
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(ROOT_DIR, 'weights', model_name + '.pth')

    if not os.path.isfile(model_path):
        for url in file_url:
            model_path = load_file_from_url(
                url=url,
                model_dir=os.path.join(ROOT_DIR, 'weights'),
                progress=True
            )

    dni_weight = None
    # if model_name == 'realesr-general-x4v3' and denoise_strength != 1:
    #     wdn_model_path = model_path.replace(
    #         'realesr-general-x4v3',
    #         'realesr-general-wdn-x4v3'
    #     )
    #     model_path = [model_path, wdn_model_path]
    #     dni_weight = [denoise_strength, 1 - denoise_strength]

    _upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=not fp32,
        gpu_id=gpu_id
    )

    return _upsampler


def sr_image_np(img: np.ndarray, outscale=4) -> np.ndarray:
    if img is None:
        raise ValueError("Input image is None")

    upsampler = _init_upsampler()

    try:
        output, _ = upsampler.enhance(img, outscale=outscale)
    except RuntimeError as e:
        raise RuntimeError(
            f"Real-ESRGAN inference failed: {e}"
        )

    return output
