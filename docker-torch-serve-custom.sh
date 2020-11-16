docker run --rm --shm-size=1g \
        --name torch-serve \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -p 8080:8080 \
        -p 8081:8081 \
        --mount type=bind,source=/home/gatvprojects/Desktop/plate_detection/pytorch-serving-detectron2/model_store,target=/tmp/models \
        vgarcia96/docker:torchserve-detectron2-cpu-1.0.0 \
        torchserve --model-store=/tmp/models
