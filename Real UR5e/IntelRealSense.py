# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 14:38:00 2021

@author: 14488
"""
import pyrealsense2 as rs
import numpy as np
import cv2

class IntelRealSense():
    def __init__(self,
                 RGB_resolution = (320,240),
                 Depth_resolution = (640,480)):
        
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, Depth_resolution[0],Depth_resolution[1], rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, RGB_resolution[0], RGB_resolution[1], rs.format.bgr8, 30)
        # Start streaming
        self.pipeline.start(self.config)
        print('IntelRealSense is connected')
        
    
    def get_rbg(self,
                resize = False,
                visulization = False,
                crop = False):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
    
        if crop:
            color_image = self.crop(crop,color_image)
            
        if resize:
            color_image = cv2.resize(color_image, resize)
            
        if visulization:
            self.visulization(color_image)
            
        return color_image
    
    def visulization(self,
                     image,
                     Window_name = 'RealSense',
                     Window_size = (640, 480)):
        cv2.namedWindow(Window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(Window_name, Window_size[0], Window_size[1])
        print("Press esc or 'q' to close the image window")
        while True:
            cv2.imshow(Window_name, image)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
            
    def crop(self,
             size,
             image):
        image_cropped = image[size[0]:size[1], size[2]:size[3]]
        return image_cropped
        
        
        
        