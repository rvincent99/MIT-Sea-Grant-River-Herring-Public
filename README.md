# MIT-Sea-Grant-River-Herring-Public
Code and supporting information for the MIT Sea Grant machine learning automated river herring video monitoring system  


## Quick start - Run pre-trained model  
(1) Setup environment: `conda env create -f environment.yml && conda activate fish-ai`  
(2) Download weights  to `/models`  
(3) Put video files in `data/raw/videos/` and list them in `examples/video_list.txt`  
(4) Run: `bash scripts/inference.sh examples/video_list.txt`  


## Quick start - Train new model  
(1) Annotate in CVAT.ai → export **COCO**  
(2) Organize the images and labels into train/, val/, and test/ folders under `data/training`.  
(3) YOLO model training  
(4) Inference with the `best.pt`. Run: `bash scripts/inference.sh examples/video_list.txt`  


