#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""


def add_custom_config(_C):
    # Add your own customized configs.
    _C.MODEL.PRE_TRAINED = None
    _C.MODEL.TWO_STREAM = False
    _C.DATA.DELETE = [
        "d18ef16d-f803-4387-bb5e-7876f1522a63",
        "77ed1624-f87b-4196-9a0a-95b7023b18e4",
        "128f69dd-f61d-494f-84c2-f3a8c937ae8f",
        "09910d48-41f9-4a84-bb1b-1bc96de62286",
        "8e914832-2dd1-44fd-81f8-1b7e2ccd2402",
    ]
    _C.MODEL.FLOW_ONLY = False
    _C.TRAIN.MODE = "train"
    pass
