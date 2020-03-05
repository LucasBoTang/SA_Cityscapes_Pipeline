#!/usr/bin/env python
# coding: utf-8

import os
import argparse

from utils import createPanopticImgs, evalPanopticSemanticLabeling
from PATH import *

from cityscapesscripts.helpers.csHelpers import printError

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_fdr', type=str, default="prob_heur")
    args = parser.parse_args()

    gt_in_path = DATA_PATH + "/gtFine"
    gt_out_path = DATA_PATH + "/panoptic"

    pred_in_path = "./experiments_eccv/" + args.exp_fdr
    pred_out_path = pred_in_path + "/panoptic"

    gt_jsonfile = gt_out_path + "/cityscapes_panoptic_val_trainId.json"
    gt_folder = gt_out_path + "/cityscapes_panoptic_val_trainId"
    pred_jsonfile = pred_out_path + "/cityscapes_panoptic_val_trainId.json"
    pred_folder = pred_out_path + "/cityscapes_panoptic_val_trainId"
    result = pred_in_path + "/result.json"

    if not os.path.isfile(gt_jsonfile):
        if not os.path.isdir(gt_out_path):
            os.mkdir(gt_out_path)
        createPanopticImgs.convert2panoptic(gt_in_path, gt_out_path, True)

    if not os.path.isfile(pred_jsonfile):
        if not os.path.isdir(pred_out_path):
            os.mkdir(pred_out_path)
        createPanopticImgs.convert2panoptic(pred_in_path, pred_out_path, True)

    cityscapesPath = os.environ.get(
        'CITYSCAPES_DATASET', os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')
    )
    gtJsonFile = os.path.join(cityscapesPath, "gtFine", "cityscapes_panoptic_val.json")

    predictionPath = os.environ.get(
        'CITYSCAPES_RESULTS',
        os.path.join(cityscapesPath, "results")
    )
    predictionJsonFile = os.path.join(predictionPath, "cityscapes_panoptic_val.json")

    if not os.path.isfile(gt_jsonfile):
        printError("Could not find a ground truth json file in {}. Please run the script with '--help'".format(gt_jsonfile))
    if gt_folder is None:
        gt_folder = os.path.splitext(gt_jsonfile)[0]

    if not os.path.isfile(pred_jsonfile):
        printError("Could not find a prediction json file in {}. Please run the script with '--help'".format(pred_jsonfile))
    if pred_folder is None:
        pred_folder = os.path.splitext(pred_jsonfile)[0]

    evalPanopticSemanticLabeling.evaluatePanoptic(gt_jsonfile, gt_folder, pred_jsonfile, pred_folder, result)
