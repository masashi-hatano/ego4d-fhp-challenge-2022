import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from slowfast.utils.parser import load_config
import slowfast.utils.checkpoint as cu
import slowfast.utils.logging as logging
from slowfast.datasets import loader
from slowfast.models import build_model
from utils.util import scaling


logger = logging.get_logger(__name__)


@torch.no_grad()
def evaluation_val(cfg, model, loader, num_vis, outputdir, plot=False):
    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()

    left_list = []
    right_list = []
    left_final_list = []
    right_final_list = []

    for cur_iter, (inputs, labels, masks, _, meta) in enumerate(loader):
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            labels = labels.cuda()
            masks = masks.cuda()
        if cfg.MODEL.PRE_TRAINED:
            inputs = inputs[0]

        preds = model(inputs)
        preds = torch.mul(preds, masks).cuda()

        # adjust the scale
        preds = scaling(cfg, preds, meta[2:4])

        # Cuda -> CPU
        preds = preds.to("cpu").detach().numpy()
        labels = labels.to("cpu").detach().numpy()

        right_list, left_list, right_final_list, left_final_list = calculate_L2_loss(
            preds,
            labels,
            right_list,
            left_list,
            right_final_list,
            left_final_list,
        )

        if plot and cur_iter < num_vis:
            visualize_val(preds, labels, masks, meta, outputdir)

    print(
        "***left hand mean disp error {:.3f}, right hand mean disp error {:.3f}".format(
            sum(left_list) / len(left_list), sum(right_list) / len(right_list)
        )
    )
    print(
        "***left hand contact disp error {:.3f}, right hand contact disp error {:.3f}".format(
            sum(left_final_list) / len(left_final_list),
            sum(right_final_list) / len(right_final_list),
        )
    )


@torch.no_grad()
def evaluation_test(cfg, model, loader, num_vis, outputdir, plot=False):
    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()

    for cur_iter, (inputs, labels, masks, _, meta) in enumerate(loader):
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            labels = labels.cuda()
            masks = masks.cuda()
        if cfg.MODEL.PRE_TRAINED:
            inputs = inputs[0]

        preds = model(inputs)

        # adjust the scale
        preds = scaling(cfg, preds, meta[2:4])

        # Cuda -> CPU
        preds = preds.to("cpu").detach().numpy()
        labels = labels.to("cpu").detach().numpy()

        if plot and cur_iter < num_vis:
            visualize_test(preds, meta, outputdir)
        else:
            break


def calculate_L2_loss(
    preds, labels, right_list, left_list, right_final_list, left_final_list
):
    for pred, label in zip(preds, labels):
        for k in range(5):
            l_x_pred = pred[k * 4]
            l_y_pred = pred[k * 4 + 1]
            r_x_pred = pred[k * 4 + 2]
            r_y_pred = pred[k * 4 + 3]

            l_x_gt = label[k * 4]
            l_y_gt = label[k * 4 + 1]
            r_x_gt = label[k * 4 + 2]
            r_y_gt = label[k * 4 + 3]

            if r_x_gt != 0 or r_y_gt != 0:
                dist = np.sqrt((r_y_gt - r_y_pred) ** 2 + (r_x_gt - r_x_pred) ** 2)
                right_list.append(dist)
            if l_x_gt != 0 or l_y_gt != 0:
                dist = np.sqrt((l_y_gt - l_y_pred) ** 2 + (l_x_gt - l_x_pred) ** 2)
                left_list.append(dist)

        if r_x_gt != 0 or r_y_gt != 0:
            dist = np.sqrt((r_y_gt - r_y_pred) ** 2 + (r_x_gt - r_x_pred) ** 2)
            right_final_list.append(dist)
        if l_x_gt != 0 or l_y_gt != 0:
            dist = np.sqrt((l_y_gt - l_y_pred) ** 2 + (l_x_gt - l_x_pred) ** 2)
            left_final_list.append(dist)

    return right_list, left_list, right_final_list, left_final_list


def visualize_val(preds, labels, masks, meta, outputdir):
    # Use only the first item of the batch
    pred = preds[0]
    label = labels[0]
    mask = masks[0]
    input_dir = Path(meta[0][0])
    pre45_frame = int(meta[1][0])
    outputdir = Path(outputdir)

    print(f"pred: {pred}")
    print(f"label: {label}")

    for key_frame in range(5):
        frame = pre45_frame + 15 * key_frame
        input_path = input_dir / Path(str(frame).zfill(6))
        img = cv2.imread(str(input_path) + ".png")
        for single_hand in range(2):
            if mask[key_frame * 4 + single_hand * 2]:
                cd_pred = (
                    int(pred[key_frame * 4 + single_hand * 2]),
                    int(pred[key_frame * 4 + single_hand * 2 + 1]),
                )
                cd_label = (
                    int(label[key_frame * 4 + single_hand * 2]),
                    int(label[key_frame * 4 + single_hand * 2 + 1]),
                )
                cv2.circle(img, cd_pred, 5, (255, 0, 0), thickness=-1)
                cv2.circle(img, cd_label, 5, (0, 255, 0), thickness=-1)
        video_name = input_dir.name
        output_sudir = outputdir / Path("visualization/val") / video_name
        output_sudir.mkdir(parents=True, exist_ok=True)
        output_path = output_sudir / Path(str(frame).zfill(6))
        cv2.imwrite(str(output_path) + ".png", img)


def visualize_test(preds, meta, outputdir):
    # Use only the first item of the batch
    pred = preds[0]
    input_dir = Path(meta[0][0])
    pre45_frame = int(meta[1][0])
    outputdir = Path(outputdir)

    print(f"pred: {pred}")

    for key_frame in range(5):
        frame = pre45_frame + 15 * key_frame
        input_path = input_dir / Path(str(frame).zfill(6))
        img = cv2.imread(str(input_path) + ".png")
        for single_hand in range(2):
            cd_pred = (
                int(pred[key_frame * 4 + single_hand * 2]),
                int(pred[key_frame * 4 + single_hand * 2 + 1]),
            )
            cv2.circle(img, cd_pred, 5, (255, 0, 0), thickness=-1)
        video_name = input_dir.name
        output_sudir = outputdir / Path("visualization/test") / video_name
        output_sudir.mkdir(parents=True, exist_ok=True)
        output_path = output_sudir / Path(str(frame).zfill(6))
        cv2.imwrite(str(output_path) + ".png", img)


def main(cfg, num_vis, eval_type, plot):
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Build the video model and print model statistics.
    model = build_model(cfg)

    # Create the video loader.
    assert eval_type in ["val", "test"]
    video_loader = loader.construct_loader(cfg, eval_type)

    # Load checkpoint.
    if cfg.TRAIN.CHECKPOINT_FILE_PATH is not None:
        checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
    elif cu.has_checkpoint(cfg.OUTPUT_DIR):
        checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
    else:
        logger.info("Find no checkpoint file")
    logger.info("Load from {}".format(checkpoint))
    cu.load_checkpoint(checkpoint, model, cfg.NUM_GPUS > 1)

    # Evaluation
    if eval_type == "val":
        evaluation_val(cfg, model, video_loader, num_vis, cfg.OUTPUT_DIR, plot=plot)
    elif eval_type == "test":
        evaluation_test(cfg, model, video_loader, num_vis, cfg.OUTPUT_DIR, plot=plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", default="path_to_config")
    parser.add_argument("--opts", default=None)
    parser.add_argument("--num_vis", default=10)
    parser.add_argument("--plot", default=True)
    parser.add_argument("--eval_type", default="val")  # val or test
    args = parser.parse_args()
    cfg = load_config(args)
    main(cfg, args.num_vis, args.eval_type, args.plot)
