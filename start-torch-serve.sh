#!/bin/bash
torchserve --start --model-store model_store --models plate-detector=plate_detector.mar
