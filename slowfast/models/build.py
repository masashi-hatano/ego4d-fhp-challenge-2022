#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Model construction functions."""

import torch
import torch.nn as nn
from pytorchvideo.models.head import create_res_basic_head
from fvcore.common.registry import Registry

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for video model.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""


def build_model(cfg, gpu_id=None):
    """
    Builds the video model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in slowfast/config/defaults.py.
        gpu_id (Optional[int]): specify the gpu index to build model.
    """
    if torch.cuda.is_available():
        assert (
            cfg.NUM_GPUS <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
            cfg.NUM_GPUS == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    # Construct the model
    if cfg.MODEL.PRE_TRAINED is not None:
        model = _construct_model(cfg)
    else:
        name = cfg.MODEL.MODEL_NAME
        model = MODEL_REGISTRY.get(name)(cfg)

    if cfg.NUM_GPUS:
        if gpu_id is None:
            # Determine the GPU used by the current process
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id
        # Transfer the model to the current GPU device
        model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device
        )
    return model


def _construct_model(cfg):
    if cfg.MODEL.PRE_TRAINED == "i3d_r50":
        model = torch.hub.load(
            "facebookresearch/pytorchvideo",
            model=cfg.MODEL.PRE_TRAINED,
            pretrained=True,
        )
        model.blocks[6] = create_res_basic_head(
            in_features=cfg.RESNET.WIDTH_PER_GROUP * 32,
            out_features=cfg.MODEL.NUM_CLASSES,
            pool=nn.AvgPool3d,
            output_size=(1, 1, 1),
            pool_kernel_size=(4, 7, 7),
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            activation=None,
            output_with_global_average=True,
        )
    else:
        raise Exception("Only supported for isd_r50.")
    return model
