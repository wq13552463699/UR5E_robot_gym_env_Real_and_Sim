#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 05:43:47 2021

@author: wq
"""

import os
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
import numpy as np
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from numpy import mat

# 以下这部分是你自己要编辑的部分，根据你的机器人的配置去校准的环境的四个角的坐标
rob_cor_1 = (0.337180175851907, -0.7709528989764918)
rob_cor_2 = (-0.3383507457068013, -0.7918474781347146)
rob_cor_3 = (0.3435026039288244, -0.3769407945516401)
rob_cor_4 = (-0.3350733477311105, -0.3822064940321181)
################################################################################

def get_metadata():
    path_to_train_image = './trained_cnn/UR5_sim_coco/train'
    path_to_train_json = './trained_cnn/UR5_sim_coco/annotations/train.json'
    register_coco_instances(
        'train', {}, path_to_train_json, path_to_train_image)
    coco_val_metadata = MetadataCatalog.get('train')
    return coco_val_metadata

def get_predictor():
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu"
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # Let training initialize from model zoo
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "./trained_cnn/model_final.pth")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # only has one class (ballon).
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    return predictor

def brg2rgb(image_rgb, resolution):
    image_rgb_r = [image_rgb[i] for i in range(0, len(image_rgb), 3)]
    image_rgb_r = np.array(image_rgb_r)
    image_rgb_r = image_rgb_r.reshape(resolution[1], resolution[0])
    image_rgb_r = image_rgb_r.astype(np.uint8)
    image_rgb_g = [image_rgb[i] for i in range(1, len(image_rgb), 3)]
    image_rgb_g = np.array(image_rgb_g)
    image_rgb_g = image_rgb_g.reshape(resolution[1], resolution[0])
    image_rgb_g = image_rgb_g.astype(np.uint8)
    image_rgb_b = [image_rgb[i] for i in range(2, len(image_rgb), 3)]
    image_rgb_b = np.array(image_rgb_b)
    image_rgb_b = image_rgb_b.reshape(resolution[1], resolution[0])
    image_rgb_b = image_rgb_b.astype(np.uint8)
    result_rgb = cv2.merge([image_rgb_b, image_rgb_g, image_rgb_r])
    result_rgb = cv2.flip(result_rgb, 0)
    return result_rgb

def visulization(result_rgb, metadata, outputs):
    v = Visualizer(result_rgb[:, :, ::-1],metadata=metadata, scale=0.5, instance_mode=ColorMode.IMAGE_BW)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.namedWindow("prediction",0);
    cv2.resizeWindow("prediction", 1024, 512)
    cv2.moveWindow("prediction",0,0)
    cv2.imshow("prediction",out.get_image()[:, :, ::-1])
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
    
from copy import copy

def loc_label(outputs):

    boxes = outputs["instances"].pred_boxes
    center_pos = boxes.get_centers()
    result_pos = center_pos.numpy().tolist()
    sorted_list = []
    result_pocy = copy(result_pos)
    
    for i in range(len(result_pos)):  
        resmin = result_pocy[0]
        for j in range(len(result_pocy)):
            if resmin[0] > result_pocy[j][0]:
                resmin = result_pocy[j]
        sorted_list.append(resmin)
        result_pocy.remove(resmin)
        
    label1_3 = [sorted_list[0],sorted_list[1]]
    label2_4 = [sorted_list[-1],sorted_list[-2]]
    
    if label1_3[0][1] < label1_3[1][1]:
        label1 = label1_3[0]
        label3 = label1_3[1]
    else:
        label1 = label1_3[1]
        label3 = label1_3[0]

    if label2_4[0][1] < label2_4[1][1]:
        label2 = label2_4[0]
        label4 = label2_4[1]
    else:
        label2 = label2_4[1]
        label4 = label2_4[0]
        
    return [label1, label2, label3, label4]

def cal_obj_pos(obj_rgb_coor,label_coordinate):
    
    rgb_cor_1 = label_coordinate[0]
    rgb_cor_2 = label_coordinate[1]
    rgb_cor_3 = label_coordinate[2]
    rgb_cor_4 = label_coordinate[3]

    dy_rob_1 = rob_cor_3[1] - rob_cor_1[1]
    dy_rob_2 = rob_cor_4[1] - rob_cor_2[1]
    dx_rob_1 = rob_cor_2[0] - rob_cor_1[0]
    dx_rob_2 = rob_cor_4[0] - rob_cor_3[0]
    
    dy_rgb_1 = rgb_cor_3[1] - rgb_cor_1[1]
    dy_rgb_2 = rgb_cor_4[1] - rgb_cor_2[1]
    dx_rgb_1 = rgb_cor_2[0] - rgb_cor_1[0]
    dx_rgb_2 = rgb_cor_4[0] - rgb_cor_3[0]
    
    obj_x_1 = (((obj_rgb_coor[0] - rgb_cor_1[0]) / dx_rgb_1) * dx_rob_1) + rob_cor_1[0]
    obj_x_2 = (((obj_rgb_coor[0] - rgb_cor_2[0]) / dx_rgb_2) * dx_rob_2) + rob_cor_2[0]
    obj_x = (obj_x_1 + obj_x_2) / 2
    # print('x coordinate in the robot coordinate system is: ', obj_x)
    obj_y_1 = (((obj_rgb_coor[1] - rgb_cor_1[1]) / dy_rgb_1) * dy_rob_1) + rob_cor_1[1]
    obj_y_2 = (((obj_rgb_coor[1] - rgb_cor_2[1]) / dy_rgb_2) * dy_rob_2) + rob_cor_2[1]
    obj_y = (obj_y_1 + obj_y_2) / 2
    # print('y coordinate in the robot coordinate system is: ', obj_y)
    return (obj_x, obj_y)


def get_all_objects_coordinate(cubiod_coor,sphere_coor,label_coordinate):
    cub_coor = []
    sph_coor = []
    for cub in cubiod_coor:
        cub_coor.append(cal_obj_pos(cub,label_coordinate))
    for sph in sphere_coor:
        sph_coor.append(cal_obj_pos(sph,label_coordinate))
    return cub_coor, sph_coor

def list2mat(list):
    m1 = [list[0],list[1],list[2],list[3]]
    m2 = [list[4],list[5],list[6],list[7]]
    m3 = [list[8],list[9],list[10],list[11]]
    m4 = [list[12],list[13],list[14],list[15]]
    matrix = mat([m1,m2,m3,m4])
    return matrix
            
def mat2list(matrix):
    lis = [matrix[0,0],matrix[0,1],matrix[0,2],matrix[0,3],\
           matrix[1,0],matrix[1,1],matrix[1,2],matrix[1,3],\
           matrix[2,0],matrix[2,1],matrix[2,2],matrix[2,3],\
           matrix[3,0],matrix[3,1],matrix[3,2],matrix[3,3]]
    return lis
            
            
            