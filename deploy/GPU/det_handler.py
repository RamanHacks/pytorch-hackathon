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
import logging

logger = logging.getLogger(__name__)

class DetectionHandler(BaseHandler):

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
        # config = {"job_name": "det_infer", "level": "INFO", "batch_size": 1, "device": "cpu", "input": "test/", "output": "test_output/", "model_location": "/home/fsmlp/Desktop/MyRepo/UCR/models/det/det_vgg_craft.pt", "Preprocess": [{"DetResizeForTest": {"resize_long": 1920}}, {"NormalizeImage": {"scale": "1./255.", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "order": "hwc"}}, {"ToCHWImage": None}, {"KeepKeys": {"keep_keys": ["image", "shape"]}}], "Architecture": {"model_type": "det", "algorithm": "CRAFT", "Transform": None, "Backbone": {"name": "VGG16_BN"}, "Neck": {"name": "CRAFTFPN"}, "Head": {"name": "CRAFTHead"}}, "Postprocess": {"name": "CRAFTPostProcess", "text_thresh": 0.35, "link_thresh": 0.1, "min_size": 3, "use_dilate": False, "xdilate": 9, "ydilate": 3, "xpad": 4, "ypad": 4, "rotated_box": True}}
        config = {"job_name": "det_infer", "level": "INFO", "batch_size": 1, "device": "cuda", "input": "test/", "output": "test_output/", "model_location": "/home/fsmlp/Desktop/MyRepo/UCR/models/det/det_vgg_craft.pt", "Preprocess": [{"DetResizeForTest": {"image_shape": (960,736)}}, {"NormalizeImage": {"scale": "1./255.", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "order": "hwc"}}, {"ToCHWImage": None}, {"KeepKeys": {"keep_keys": ["image", "shape"]}}], "Architecture": {"model_type": "det", "algorithm": "CRAFT", "Transform": None, "Backbone": {"name": "VGG16_BN"}, "Neck": {"name": "CRAFTFPN"}, "Head": {"name": "CRAFTHead"}}, "Postprocess": {"name": "CRAFTPostProcess", "text_thresh": 0.35, "link_thresh": 0.1, "min_size": 3, "use_dilate": False, "xdilate": 9, "ydilate": 3, "xpad": 4, "ypad": 4, "rotated_box": True}}
        
        self.preprocess_op = build_preprocess(config["Preprocess"])
        self.postprocess_op = build_postprocess(config["Postprocess"])
        
        self.initialized = True
        
        
    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        images = []
        shape_list = []
        img_shapes = []
        for row in data:
            input = row.get("data") or row.get("body")
            image = base64.b64decode(input)
            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image)).convert('RGB')
                cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                img_shapes.append(cv2_image.shape)
                data = {"image": cv2_image}
                img, shape = preprocess(data, self.preprocess_op)

            images.append(torch.tensor(img))
            shape_list.append(np.expand_dims(shape, axis=0))
        return torch.stack(images).to(self.device), shape_list, img_shapes

    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points
    
    def sorted_boxes(self, dt_boxes):
        """
        Sort text boxes in order from top to bottom, left to right
        config:
            dt_boxes(array):detected text boxes with shape [4, 2]
        return:
            sorted boxes(array) with shape [4, 2]
        """
        dt_boxes = np.array(dt_boxes).astype('float32')
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        # _boxes = list(sorted_boxes)
        _boxes = np.asarray(sorted_boxes).tolist()

        for i in range(num_boxes - 1):
            if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and (
                _boxes[i + 1][0][0] < _boxes[i][0][0]
            ):
                tmp = _boxes[i]
                _boxes[i] = _boxes[i + 1]
                _boxes[i + 1] = tmp
        return _boxes        
        
    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box.tolist())
        # return(self.sorted_boxes(dt_boxes_new))
        return dt_boxes_new
    
    def postprocess(self, data, shape_list, img_shapes):     
        outputs = []
        for output_tensor in data:
            output = output_tensor.cpu().data.numpy()
            outputs.append(output)
            
        out_list = []
        for i in range(len(data[0])):            
            preds={}
            
            preds["text_map"] = np.expand_dims(outputs[0][i,:,:], axis=0)
            preds["link_map"] = np.expand_dims(outputs[1][i,:,:], axis=0)
            post_result = self.postprocess_op(preds, shape_list[i])
            
            dt_boxes = post_result[0]["points"]
            dt_boxes = self.filter_tag_det_res(dt_boxes, img_shapes[i])
            
            out_list.append(dt_boxes)
        
        return out_list
    
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
        images, shape_list, img_shapes = self.preprocess(data)
        model_output = self.inference(images)
        output =  self.postprocess(model_output, shape_list, img_shapes)
        return output