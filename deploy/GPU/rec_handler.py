from ts.torch_handler.base_handler import BaseHandler
import torch
import cv2
import os
import io 
import base64
import numpy as np
from PIL import Image
from ucr.core.preprocess import build_preprocess, preprocess
from ucr.core.postprocess import build_postprocess
from ucr.core.preprocess.label_ops import BaseRecLabelEncode
import logging
import ucr
logger = logging.getLogger(__name__)

class RecHandler(BaseHandler):

    def __init__(self):
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        # load the model
        properties = context.system_properties
        self.map_location = "cuda" if torch.cuda.is_available() and properties.get("gpu_id") is not None else "cpu"
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else self.map_location
        )
        self.manifest = context.manifest

        model_dir = properties.get("model_dir")
        model_pt_path = None
        if "serializedFile" in self.manifest["model"]:
            serialized_file = self.manifest["model"]["serializedFile"]
            model_pt_path = os.path.join(model_dir, serialized_file)

        # model def file
        model_file = self.manifest["model"].get("modelFile", "")

        if model_file:
            logger.debug("Loading eager model")
            self.model = self._load_pickled_model(model_dir, model_file, model_pt_path)
            self.model.to(self.device)
        else:
            logger.debug("Loading torchscript model")
            if not os.path.isfile(model_pt_path):
                raise RuntimeError("Missing the model.pt file")

            self.model = self._load_torchscript_model(model_pt_path)

        self.model.eval()

        logger.debug('Model file %s loaded successfully', model_pt_path)
        
        # preprocess and postprocess initialization
        
        config = {"job_name": "rec_infer", "level": "INFO", "batch_size": 8, "device": "cuda", "lang": "ch_sim", "max_text_length": 25, "use_space_char": True, "whitelist": "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]_abcdefghijklmnopqrstuvwxyz{|}~\u2014\u2018\u2019\u201c\u201d", "blacklist": None, "input": "test/", "model_location": "/home/fsmlp/Desktop/MyRepo/UCR/models/rec/multi_language/rec_ch_sim_ppocr_server.pt", "font_path": None, "char_dict_location": ucr.__file__.replace('__init__.py', '') + 'utils/dict/ch_sim_dict.txt', "Preprocess": [{"RecResizeImg": {"image_shape": [3, 32, 100]}}], "Architecture": {"model_type": "rec", "algorithm": "CRNN", "Transform": None, "Backbone": {"name": "ResNet", "layers": 34}, "Neck": {"name": "SequenceEncoder", "encoder_type": "rnn", "hidden_size": 256}, "Head": {"name": "CTCHead", "fc_decay": 4e-05}}, "Postprocess": {"name": "CTCLabelDecode"}}
        for op in config["Preprocess"]:
            op_name = list(op)[0]
            if op_name == "RecResizeImg":
                image_shape = op[op_name][
                    "image_shape"
                ]  # TODO:add try except here

        self.whitelist = config["whitelist"]
        self.blacklist = config["blacklist"]
        
        char_dict_location = config["char_dict_location"]
        self.char_ops = BaseRecLabelEncode(
            config["max_text_length"],
            char_dict_location,
            config["lang"],
            config["use_space_char"],
        )

        global_keys = ["lang", "use_space_char", "max_text_length"]
        global_cfg = {
            key: value for key, value in config.items() if key in global_keys
        }
        global_cfg["char_dict_location"] = char_dict_location

        self.preprocess_op = build_preprocess(config["Preprocess"])
        self.postprocess_op = build_postprocess(
            config["Postprocess"], global_cfg
        )
        
        self.initialized = True
        
    def get_rotate_crop_image(self, img, points):
        """
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        """
        
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3]),
            )
        )
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2]),
            )
        )
        pts_std = np.float32(
            [
                [0, 0],
                [img_crop_width, 0],
                [img_crop_width, img_crop_height],
                [0, img_crop_height],
            ]
        )
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M,
            (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC,
        )
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img
        
    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        images = []
        ino_list = []
        shape_list = []
        for ino,row in enumerate(data):
            image = row.get("pre_processing").decode()
            if isinstance(image, str):
                image = base64.b64decode(image)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image)).convert('RGB')
                img_shape = image.size
                img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                bbox = eval(row.get("detection").decode())
                bbox = np.array(bbox).astype('float32')
                for bno in range(len(bbox)):
                    img_crop = self.get_rotate_crop_image(img, bbox[bno])
                    norm_img = preprocess({"image":img_crop}, self.preprocess_op)
                    images.append(torch.tensor(norm_img["image"]))
                    ino_list.append((ino,bbox[bno]))
                    shape_list.append(img_shape)
        return torch.stack(images).to(self.device), ino_list, shape_list
    
    def postprocess(self, data, ino_list, shape_list):     
        preds = data.cpu().numpy()
        
        if not self.whitelist and self.blacklist:
            self.mod_chars = np.arange(preds.shape[-1])
            black_list = self.char_ops.encode(self.blacklist)
            black_list = np.array(black_list) + 1
            self.mod_chars = np.setdiff1d(self.mod_chars, black_list)
        elif self.whitelist:
            white_list = self.char_ops.encode(self.whitelist)
            self.mod_chars = np.append(white_list, [-1]) + 1
        elif not self.whitelist and not self.blacklist:
            self.mod_chars = []

        if len(self.mod_chars) != 0:
            mod_onehot = np.zeros(preds.shape[-1])
            mod_onehot[self.mod_chars] = 1
            preds = np.multiply(
                preds, mod_onehot
            )  
            
        rec_result = self.postprocess_op(preds)
        
        out_list = []
        oldidx = ino_list[0][0]
        rec_res = []
        for rno in range(len(rec_result)):
            newidx = ino_list[rno][0]
            if oldidx != newidx:
                out_list.append(rec_res)
                oldidx = newidx
                if rec_result[rno][0] != '':
                    rec_res = [{
                        'bbox':self.normalize(ino_list[rno][1], shape_list[rno]),
                        'ocr':rec_result[rno],
                        'key': ['#other', 1]
                        }]
                else:
                    rec_res = []
            else:
                if rec_result[rno][0] != '':
                    rec_res.append({
                        'bbox':self.normalize(ino_list[rno][1], shape_list[rno]),
                        'ocr':rec_result[rno],
                        'key': ['#other', 1]
                        })

        out_list.append(rec_res)
        return out_list
    
    def normalize(self, bbox, shape):
        shape_array = np.array(shape*4)
        return (bbox.flatten()/shape_array).reshape(4,2).tolist()
        
    
    def handle(self, data, context):
        """Entry point for default handler. It takes the data from the input request and returns
           the predicted outcome for the input.
        Args:
            data (list): The input data that needs to be made a prediction request on.
            context (Context): It is a JSON Object containing information pertaining to
                               the model artefacts parameters.
        Returns:
            list : Returns a list of dictionary with the predicted response.
        """
        self.context = context

        images, ino_list, shape_list = self.preprocess(data)
        model_output = self.inference(images)
        output =  self.postprocess(model_output, ino_list, shape_list)
        return output