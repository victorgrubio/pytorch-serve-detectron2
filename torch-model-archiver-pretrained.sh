torch-model-archiver -f --model-name plate_detector \
                     --version 1.0 \
                     --serialized-file serialized_models/model_plate_detection.pth \
                     --export-path model_store \
                     --model-file ../detectron2/detectron2/engine/defaults.py \
                     --extra-files ../detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
                     --handler object_detector

