import logging
import torch
import torch.nn.functional as F
import io
import os
import traceback
import numpy as np
from PIL import Image
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

class CustomHandler(BaseHandler):
    """
    Custom handler for pytorch serve. This handler supports batch requests.
    For a deep description of all method check out the doc:
    https://pytorch.org/serve/custom_service.html
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = None
        self.mapping = None
        self.device = False
        self.initialized = False

    def initialize(self, ctx):
        properties = ctx.system_properties
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        model_dir = properties.get("model_dir")
        # Read model serialize/pth file
        self.model = self.load_detectron2(model_dir)
        self.initialized = True
    
    def load_detectron2(self, model_dir):
        cfg = get_cfg()
        cfg.OUTPUT_DIR = model_dir # Modify with the folder path you want
        cfg.merge_from_file(os.path.join(cfg.OUTPUT_DIR, "plate_detection_config.yaml"))
        cfg.MODEL.MASK_ON = False
        cfg.MODEL.RESNETS.DEPTH = 50
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_plate_detection.pth")
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
        cfg.MODEL.DEVICE = "cpu"
        predictor = DefaultPredictor(cfg)
        return predictor

    def preprocess(self, req):
        """
        Process one single image.
        """
        image = None
        image = req[0].get("data")
        if image is None:
            request_image = req[0].get("File")
            image = np.array(Image.open(io.BytesIO(request_image)))
        return image

    def inference(self, image):
        """
        Given the data from .preprocess, perform inference using the model.
        We return the predicted label for each image.
        """
        preds = self.model(image)["instances"]
        return preds

    def postprocess(self, preds):
        """
        Given the data from .inference, postprocess the output.
        In our case, we get the human readable label from the mapping 
        file and return a json. Keep in mind that the reply must always
        be an array since we are returning a batch of responses.
        """
        res = []
        scores = preds.get("scores").tolist()
        pred_classes = preds.get("pred_classes").tolist()
        pred_boxes = preds.get("pred_boxes").tensor.tolist()
        for i in range(0, len(scores)):
            res.append({'class': pred_classes[i], 'boxes': pred_boxes[i], 'scores': scores[i]})
        return res
