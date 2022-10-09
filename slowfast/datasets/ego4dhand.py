#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data
# from iopath.common.file_io import PathManager
from iopath.common.file_io import g_pathmgr
import slowfast.utils.logging as logging

from . import decoder as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Ego4dhand(torch.utils.data.Dataset):
    """
    Ego4D video loader. Construct the ego4d video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Construct the Ego4D video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
            "trainval",
        ], "Split '{}' not supported for Ego4D Hand Anticipation".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val", "trainval"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing Ego4D {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "v1/annotations/fho_hands_{}.json".format(self.mode)
        )
        assert g_pathmgr.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )
        datadir = Path(self.cfg.DATA.PATH_TO_DATA_DIR)
        self._path_to_ant_videos = []
        self._pre45_clip_frames = []
        self._pre45_frames = []
        self._labels = []
        self._labels_masks = []
        self._clip_idx = []
        self._spatial_temporal_idx = []
        f = open(path_to_file)
        data = json.load(f)
        f.close()
        # print(self._num_clips)
        for clip_dict in data["clips"]:
            video_uid = clip_dict["video_uid"]
            if video_uid in self.cfg.DATA.DELETE:
                print(f"{video_uid} is invalid video, so it will not be included in the dataset")
                continue
            for frame_dict in clip_dict["frames"]:
                self._clip_idx.append(clip_dict["clip_id"])
                clip_uid = clip_dict["clip_uid"]
                path_to_image_frame = datadir/ Path("v1/image_frame") / Path(clip_uid)
                self._path_to_ant_videos.append(path_to_image_frame)
                self._pre45_clip_frames.append(frame_dict["pre_45"]["clip_frame"])
                self._pre45_frames.append(frame_dict["pre_45"]["frame"])
                label=[]
                label_mask=[]
                #placeholder for the 1x20 hand gt vector (padd zero when GT is not available)
                # 5 frames have the following order: pre_45, pre_40, pre_15, pre, contact
                # GT for each frames has the following order: left_x,left_y,right_x,right_y
                label= [0.0]*20
                label_mask = [0.0]*20
                if self.mode in ["train", "val", "trainval"]:
                    for frame_type, frame_annot in frame_dict.items():
                        if frame_type in [
                            'action_start_sec', 
                            'action_end_sec', 
                            'action_start_frame',
                            'action_end_frame',
                            'action_clip_start_sec', 
                            'action_clip_end_sec', 
                            'action_clip_start_frame',
                            'action_clip_end_frame',
                            ]:
                            continue
                        if frame_type == 'pre_45':
                            for single_hand in frame_annot["boxes"]:
                                if 'left_hand' in single_hand:
                                    label_mask[0]=1.0
                                    label_mask[1]=1.0
                                    label[0]= single_hand['left_hand'][0]
                                    label[1]= single_hand['left_hand'][1]
                                if 'right_hand' in single_hand:
                                    label_mask[2]=1.0
                                    label_mask[3]=1.0
                                    label[2]= single_hand['right_hand'][0]
                                    label[3]= single_hand['right_hand'][1]   
                        if frame_type == 'pre_30':
                            for single_hand in frame_annot["boxes"]:
                                if 'left_hand' in single_hand:
                                    label_mask[4]=1.0
                                    label_mask[5]=1.0
                                    label[4]= single_hand['left_hand'][0]
                                    label[5]= single_hand['left_hand'][1]
                                if 'right_hand' in single_hand:
                                    label_mask[6]=1.0
                                    label_mask[7]=1.0
                                    label[6]= single_hand['right_hand'][0]
                                    label[7]= single_hand['right_hand'][1]   
                        if frame_type == 'pre_15':
                            for single_hand in frame_annot["boxes"]:
                                if 'left_hand' in single_hand:
                                    label_mask[8]=1.0
                                    label_mask[9]=1.0
                                    label[8]= single_hand['left_hand'][0]
                                    label[9]= single_hand['left_hand'][1]
                                if 'right_hand' in single_hand:
                                    label_mask[10]=1.0
                                    label_mask[11]=1.0
                                    label[10]= single_hand['right_hand'][0]
                                    label[11]= single_hand['right_hand'][1]   
                        if frame_type == 'pre_frame':
                            for single_hand in frame_annot["boxes"]:
                                if 'left_hand' in single_hand:
                                    label_mask[12]=1.0
                                    label_mask[13]=1.0
                                    label[12]= single_hand['left_hand'][0]
                                    label[13]= single_hand['left_hand'][1]
                                if 'right_hand' in single_hand:
                                    label_mask[14]=1.0
                                    label_mask[15]=1.0
                                    label[14]= single_hand['right_hand'][0]
                                    label[15]= single_hand['right_hand'][1]    
                        if frame_type == 'contact_frame':
                            for single_hand in frame_annot["boxes"]:
                                if 'left_hand' in single_hand:
                                    label_mask[16]=1.0
                                    label_mask[17]=1.0
                                    label[16]= single_hand['left_hand'][0]
                                    label[17]= single_hand['left_hand'][1]
                                if 'right_hand' in single_hand:
                                    label_mask[18]=1.0
                                    label_mask[19]=1.0
                                    label[18]= single_hand['right_hand'][0]
                                    label[19]= single_hand['right_hand'][1]
                self._labels.append(label)
                self._labels_masks.append(label_mask)
        logger.info(
            "Constructing Ego4D dataloader (size: {})".format(
                len(self._pre45_clip_frames)
            )
        )

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train", "val", "trainval"]:
            # -1 indicates random sampling.
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["test"]:
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )
       
        # create frame name list 
        num_frames = self.cfg.DATA.NUM_FRAMES
        frame_names = list(reversed([max(1, self._pre45_clip_frames[index]-15*i) for i in range(1,num_frames+1)]))
        # load frames
        input_dir_rgb = self._path_to_ant_videos[index]
        input_dir_flow = Path(str(input_dir_rgb).replace("image_frame", "optical_flow"))
        pre45_clip_frame = self._pre45_clip_frames[index]
        input_path_rgb = input_dir_rgb / Path(str(pre45_clip_frame).zfill(6))
        input_path_flow = input_dir_flow / Path("npy") / Path(str(pre45_clip_frame).zfill(6))
        img = cv2.imread(str(input_path_rgb)+".png")
        h, w, _ = img.shape
        frames = torch.zeros(num_frames, 224, 224, 3)
        flows = torch.zeros(num_frames, 224, 224, 2)
        for i, frame in enumerate(frame_names):
            input_path_rgb = input_dir_rgb / Path(str(frame).zfill(6))
            input_path_flow = input_dir_flow / Path("npy") / Path(str(frame).zfill(6))
            img = cv2.imread(str(input_path_rgb)+".png").astype(np.float32)
            frames[i] = torch.from_numpy(cv2.resize(img, (224,224)))
            flows[i] = torch.from_numpy(np.load(str(input_path_flow)+".npy").astype(np.float32))
        
        # Perform color normalization.
        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        flows = flows.permute(3, 0, 1, 2)

        if self.mode in ["train", "val", "trainval"]:
            frames = utils.spatial_sampling(
                frames,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            )

        label = self._labels[index]
        label = torch.FloatTensor(label)
        mask = self._labels_masks[index]
        mask = torch.FloatTensor(mask)
        idx = (self._clip_idx[index], self._pre45_frames[index]-1)
        meta = [str(input_dir_rgb), pre45_clip_frame, h, w, idx]
        if self.cfg.MODEL.TWO_STREAM:
            inputs = [frames, flows]
        elif self.cfg.MODEL.FLOW_ONLY:
            inputs = [flows]
        else:
            inputs = utils.pack_pathway_output(self.cfg, frames)

        return inputs, label, mask, index, meta

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """

        return len(self._path_to_ant_videos)