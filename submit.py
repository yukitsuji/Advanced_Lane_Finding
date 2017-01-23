#!/usr/bin/env python3
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def get_calibration_param(image_url):
    images = glob.glob(image_url)

    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    corner = (9, 6)
    
    for image in images:
        img = mpimg.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, corner, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    print("finish get_calibration_param")
    return mtx, dist

def undistorted_image(distorted_img):
    
    undist_img = cv2.

class line():
    pass




if __name__ == "__main__":
    camera_matrix, dist_coeff = get_calibration_param('./camera_cal/calibration*.jpg')
    np.savez("./calibration.npz",mtx=camera_matrix, dist=dist_coeff)

    output = './output.mp4'
    clip1 = VideoFileClip('./project_video.mp4')
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(white_output, audio=False)
    print("process Finished. Please seee "
    
    