# Ego4D Hand Movement Prediction: Two-stream I3D

## Installation:
```shell
pip install -r requirements.txt
```

Since our method requires the same dependencies as SlowFast, we refer to the official implementation fo [SlowFast](https://github.com/facebookresearch/SlowFast/blob/main/INSTALL.md) for installation details.

## Data Preparation:

**Input**: 8 frames of RGB and optical flow at 0.5 second intervals, starting 2 seconds before the PRE_45 frame.

**Output**: 5 frames with hand positions on {p3,p2,p1,p,c}; left/right hand position format: x_l, y_l, x_r, y_r

**Note on Ground Truth**: In the dataloader, we choose pad zeros when hand ground truth is not available.

### Pre-process
You need to prepare optical flows by yourself by using [FlowFormer](https://github.com/drinkingcoder/FlowFormer-Official)

### Data-structure
The data should be organized as following:
```
Path_To_Data_Dir
|
└── annotations
│   │   fho_hands_train.json
│   │   fho_hands_val.json
│   │   fho_hands_test.json
│   │   fho_hands_trainval.json (contains all samples from training and validation set)
|
└── clips
│   │   clip_uid1.mp4
|   |   clip_uid2.mp4
|   |   ...
|
└── image_frame
│   └── clip_uid1
│   │   |   image_000001.png
|   |   |   image_000002.png
|   |   |   ...
│   └── clip_uid2
│   │   |   image_000001.png
|   |   |   image_000002.png
|   |   |   ...
│   └── ...
|
└── optical_flow
│   └── clip_uid1/npy
│   │   |   frame_id1.npy
|   |   |   frame_id2.npy
|   |   |   ...
│   └── clip_uid2/npy
│   │   |   frame_id1.npy
|   |   |   frame_id2.npy
|   |   |   ...
│   └── ...
```

## Training: 
```shell
 python tools/run_net.py --cfg /path/to/configs/Ego4D/I3D_8x8_R50.yaml 
```

## Testing: 
```shell
 python tools/run_net.py --cfg /path/to/configs/Ego4D/I3D_8x8_R50.yaml 
```

## Evaluation and Visualization: 
```shell
 python tools/evaluation.py --cfg /path/to//configs/Ego4D/I3D_8x8_R50.yaml
```
