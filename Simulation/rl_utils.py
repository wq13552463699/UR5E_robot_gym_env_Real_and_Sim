from numpy import mat
import numpy as np
import cv2
def list2mat(list):
    m1 = [list[0], list[1], list[2], list[3]]
    m2 = [list[4], list[5], list[6], list[7]]
    m3 = [list[8], list[9], list[10], list[11]]
    m4 = [list[12], list[13], list[14], list[15]]
    matrix = mat([m1, m2, m3, m4])
    return matrix


def mat2list(matrix):
    lis = [matrix[0, 0], matrix[0, 1], matrix[0, 2], matrix[0, 3], \
           matrix[1, 0], matrix[1, 1], matrix[1, 2], matrix[1, 3], \
           matrix[2, 0], matrix[2, 1], matrix[2, 2], matrix[2, 3], \
           matrix[3, 0], matrix[3, 1], matrix[3, 2], matrix[3, 3]]
    return lis

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

def visulization(result_rgb):
    cv2.namedWindow("Picture",0);
    # cv2.resizeWindow("PIcture", 640, 480)
    cv2.moveWindow("Picture",0,0)
    cv2.imshow("Picture",result_rgb)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 