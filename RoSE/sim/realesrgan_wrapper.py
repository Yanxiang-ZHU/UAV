import os
import sys
import cv2

def _add_self_path():
    # Ensure this file's parent dir is on sys.path
    this_dir = os.path.dirname(os.path.abspath(__file__))
    if this_dir not in sys.path:
        sys.path.insert(0, this_dir)

_add_self_path()

class RealESRGANWrapper:
    """Thin wrapper around inference_realesrgan.py pipeline.
    Usage:
        w = RealESRGANWrapper(model_name='RealESRGAN_x4plus', face_enhance=False, fp32=False, gpu_id=None)
        out = w.enhance(img_np, outscale=4)
    """
    def __init__(self, model_name='RealESRGAN_x4plus', model_path=None, face_enhance=False, fp32=False, gpu_id=None, tile=0, tile_pad=10, pre_pad=0):
        try:
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from basicsr.utils.download_util import load_file_from_url
            from realesrgan import RealESRGANer
        except Exception as e:
            raise ImportError(f"Real-ESRGAN dependencies missing: {e}")

        self.face_enhance = face_enhance
        self.gpu_available = torch.cuda.is_available()
        netscale = 4
        # choose model architecture by name (only implementing common ones used here)
        if model_name == 'RealESRGAN_x4plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
        elif model_name == 'RealESRGAN_x2plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
        else:
            # fallback to x4
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']

        # determine model_path
        if model_path is not None:
            mp = model_path
        else:
            mp = os.path.join(os.path.dirname(__file__), 'weights', model_name + '.pth')
            if not os.path.isfile(mp):
                # try downloading into local weights
                try:
                    mp = load_file_from_url(url=file_url[0], model_dir=os.path.join(os.path.dirname(__file__), 'weights'), progress=True, file_name=None)
                except Exception:
                    # leave mp as-is; caller will handle
                    pass

        if not os.path.isfile(mp):
            raise FileNotFoundError(f"Model file not found: {mp}")

        half = (not fp32) and self.gpu_available
        gpu_id_use = gpu_id if gpu_id is not None else (0 if self.gpu_available else None)

        self.upsampler = RealESRGANer(scale=netscale, model_path=mp, dni_weight=None, model=model, tile=tile, tile_pad=tile_pad, pre_pad=pre_pad, half=half, gpu_id=gpu_id_use)

        if self.face_enhance:
            try:
                from gfpgan import GFPGANer
                self.face_enhancer = GFPGANer(model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth', upscale=netscale, arch='clean', channel_multiplier=2, bg_upsampler=self.upsampler)
            except Exception:
                self.face_enhancer = None
        else:
            self.face_enhancer = None

    def enhance(self, img, outscale=4):
        """Enhance a single image (numpy BGR or RGB as used by OpenCV).
        Returns the enhanced numpy image.
        """
        if self.face_enhancer is not None:
            _, _, output = self.face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            return output
        else:
            res = self.upsampler.enhance(img, outscale=outscale)
            if isinstance(res, tuple) and len(res) >= 1:
                return res[0]
            return res
