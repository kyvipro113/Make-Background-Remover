import base64
import contextlib
import io
import json
import os.path as osp
import sys
import cv2
import numpy as np
from qtpy import QT_VERSION
import time
from mgmatting.MGMatting import *
from mgmatting import utils as utils_mg
import torch.nn as nn
import PIL.Image
from PIL import Image, ImageDraw

PY2 = sys.version[0] == "2"
QT4 = QT_VERSION[0] == "4"
__version__ = "5.6.0a0"
import utils
import logger

PIL.Image.MAX_IMAGE_PIXELS = None


@contextlib.contextmanager
def open(name, mode):
    assert mode in ["r", "w"]
    if PY2:
        mode += "b"
        encoding = None
    else:
        encoding = "utf-8"
    yield io.open(name, mode, encoding=encoding)
    return


class LabelFileError(Exception):
    pass

class Save_Config():
    use_mask_refine: bool
    model: nn.Module
    post_process: bool
    gray_scale: bool
    use_bkc_replace: bool
    background_color_rgb: np.array
    data_out: str

def load_save_config(use_mask_refine: bool, model_path: str, post_process: bool, gray_scale: bool, use_bkc_replace: bool, background_color_rgb: list, data_out: str):
    Save_Config.use_mask_refine = use_mask_refine
    Save_Config.post_process = post_process
    Save_Config.gray_scale = gray_scale
    Save_Config.use_bkc_replace = use_bkc_replace
    Save_Config.background_color_rgb = np.array(background_color_rgb)
    Save_Config.data_out = data_out

    Save_Config.model = networks.get_generator(encoder=CONFIG.model.arch.encoder, decoder=CONFIG.model.arch.decoder)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    Save_Config.model.load_state_dict(utils_mg.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)
    Save_Config.model.eval()

def set_mask_refine_state(use_mask_refine: bool):
    Save_Config.use_mask_refine = use_mask_refine

def set_gray_scale_state(gray_scale):
    Save_Config.gray_scale = gray_scale

class LabelFile(object):
    suffix = ".json"
    raw_file: str

    def __init__(self, filename=None):
        self.shapes = []
        self.imagePath = None
        self.imageData = None
        if filename is not None:
            self.load(filename)
        self.filename = filename

    @staticmethod
    def load_image_file(filename):
        print(f"Load_image_file: {filename}")
        try:
            image_pil = PIL.Image.open(filename)
            LabelFile.raw_file = filename
        except IOError:
            logger.error("Failed opening image file: {}".format(filename))
            return

        # apply orientation to image according to exif
        image_pil = utils.apply_exif_orientation(image_pil)

        with io.BytesIO() as f:
            ext = osp.splitext(filename)[1].lower()
            if PY2 and QT4:
                format = "PNG"
            elif ext in [".jpg", ".jpeg"]:
                format = "JPEG"
            else:
                format = "PNG"
            image_pil.save(f, format=format)
            f.seek(0)
            return f.read()

    def load(self, filename):
        keys = [
            "version",
            "imageData",
            "imagePath",
            "shapes",  # polygonal annotations
            "flags",  # image level flags
            "imageHeight",
            "imageWidth",
        ]
        shape_keys = [
            "label",
            "points",
            "group_id",
            "shape_type",
            "flags",
            "description",
            "mask",
        ]
        try:
            with open(filename, "r") as f:
                data = json.load(f)

            if data["imageData"] is not None:
                imageData = base64.b64decode(data["imageData"])
                if PY2 and QT4:
                    imageData = utils.img_data_to_png_data(imageData)
            else:
                # relative path from label file to relative path from cwd
                imagePath = osp.join(osp.dirname(filename), data["imagePath"])
                imageData = self.load_image_file(imagePath)
            flags = data.get("flags") or {}
            imagePath = data["imagePath"]
            self._check_image_height_and_width(
                base64.b64encode(imageData).decode("utf-8"),
                data.get("imageHeight"),
                data.get("imageWidth"),
            )
            shapes = [
                dict(
                    label=s["label"],
                    points=s["points"],
                    shape_type=s.get("shape_type", "polygon"),
                    flags=s.get("flags", {}),
                    description=s.get("description"),
                    group_id=s.get("group_id"),
                    mask=utils.img_b64_to_arr(s["mask"]).astype(bool)
                    if s.get("mask")
                    else None,
                    other_data={k: v for k, v in s.items() if k not in shape_keys},
                )
                for s in data["shapes"]
            ]
        except Exception as e:
            raise LabelFileError(e)

        otherData = {}
        for key, value in data.items():
            if key not in keys:
                otherData[key] = value

        # Only replace data after everything is loaded.
        self.flags = flags
        self.shapes = shapes
        self.imagePath = imagePath
        self.imageData = imageData
        self.filename = filename
        self.otherData = otherData

    @staticmethod
    def _check_image_height_and_width(imageData, imageHeight, imageWidth):
        img_arr = utils.img_b64_to_arr(imageData)
        if imageHeight is not None and img_arr.shape[0] != imageHeight:
            logger.error(
                "imageHeight does not match with imageData or imagePath, "
                "so getting imageHeight from actual image."
            )
            imageHeight = img_arr.shape[0]
        if imageWidth is not None and img_arr.shape[1] != imageWidth:
            logger.error(
                "imageWidth does not match with imageData or imagePath, "
                "so getting imageWidth from actual image."
            )
            imageWidth = img_arr.shape[1]
        return imageHeight, imageWidth

    def save(
        self,
        filename,
        shapes,
        imagePath,
        imageHeight,
        imageWidth,
        imageData=None,
        otherData=None,
        flags=None,
    ):
        if imageData is not None:
            imageData = base64.b64encode(imageData).decode("utf-8")
            imageHeight, imageWidth = self._check_image_height_and_width(
                imageData, imageHeight, imageWidth
            )
        if otherData is None:
            otherData = {}
        if flags is None:
            flags = {}
        data = dict(
            version=__version__,
            flags=flags,
            shapes=shapes,
            imagePath=imagePath,
            imageData=imageData,
            imageHeight=imageHeight,
            imageWidth=imageWidth,
        )
        for key, value in otherData.items():
            assert key not in data
            data[key] = value
        try:
            with open(filename, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.filename = filename

            ###
            points = data["shapes"][0]["points"]
            int_points =  [(int(x), int(y)) for x, y in points]
            H = int(data["imageHeight"])
            W = int(data["imageWidth"])
            mask_pil = Image.new('L', (W, H), 0)
            draw = ImageDraw.Draw(mask_pil)
            draw.polygon(int_points, outline=1, fill=255)
            mask = np.array(mask_pil)
            image_data = base64.b64decode(data["imageData"])
            np_arr = np.frombuffer(image_data, dtype=np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            alpha_pred = mask
            if Save_Config.use_mask_refine:
                image_dict = generator_tensor_dict_matrix(image=image, mask=mask)
                alpha_pred: np.array = single_inference(Save_Config.model, image_dict, post_process=Save_Config.post_process)
            if(Save_Config.gray_scale):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if Save_Config.use_bkc_replace:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                _alpha = alpha_pred / 255.0
                _alpha = np.stack([_alpha] * 3, axis=-1)
                B = np.ones_like(image) * Save_Config.background_color_rgb
                FI = _alpha * image
                BI = (1 - _alpha) * B
                I = FI + BI
                I = I.astype(np.uint8)
                I_pil = Image.fromarray(I)
                I_pil.show()
                _img_name = imagePath.split("/")[-1].split(".")[0]
                I_pil.save(Save_Config.data_out + "/" + _img_name + ".png")
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                I_pil = Image.fromarray(image)
                mask_refine_pil = Image.fromarray(alpha_pred)
                I_pil.putalpha(mask_refine_pil)

        except Exception as e:
            raise LabelFileError(e)

    @staticmethod
    def is_label_file(filename):
        return osp.splitext(filename)[1].lower() == LabelFile.suffix
    


