torch-model-archiver -f --model-name plate_detector_custom \
                     --version 1.0 \
                     --serialized-file serialized_models/model_plate_detection.pth \
                     --export-path model_store \
                     --extra-files ./custom_handler.py,./config/plate_detection_config.yaml \
                     --handler my_handler.py  \
                     -f

