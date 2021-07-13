import os
import sys
import socket
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import time



def read_TCP_image(data,Res = 640*480):
    imagenumpy = np.array(data,dtype = np.uint8)
    R1 = imagenumpy[0:Res].reshape((H,W))
    G1 = imagenumpy[Res:Res*2].reshape((H,W))
    B1 = imagenumpy[Res*2:Res*3].reshape((H,W))
    imgL = np.dstack((R1,G1,B1))
    imgL = np.rot90(imgL, 3)
    return imgL


def readnbyte(sock, n):
    buff = bytearray(n)
    pos = 0
    while pos < n:
        cr = sock.recv_into(memoryview(buff)[pos:])
        pos += cr
    return buff


def load_calibration_params(calibration_file_path):
    calibration_file = open(calibration_file_path, "rb")
    calibration_params = pickle.load(calibration_file)
    calibration_file.close()
    return calibration_params

def undistort_fisheye_no_files(calibration_params,img,side):
    start = time.time()

    K,D,DIM = calibration_params
    distorted = img
    K_new = K.copy()
    if side == 'left' or side == 'right':
        K_new[0,0] = K[0,0]/3
        K_new[1,1] = K[1,1]/3
    elif side == 'front' or side == 'back':
        K_new[0,0] = K[0,0]/2
        K_new[1,1] = K[1,1]/2

    end = time.time()

    print(f'undistort load params : {(end - start)}')

    start = time.time()

    undistorted_img = cv2.fisheye.undistortImage(distorted,K,D,None,K_new)

    end = time.time()

    print(f'undistort undistort fish opencv : {(end - start)}')
    return undistorted_img



def warp(img,DIM,side):
    X = DIM[0]
    Y = DIM[1]
    
    if side == 'left':
        #TL,TR,BR,BL
        pt1 = [237,231]
        pt2 = [396,231]
        pt3 = [550,334]
        pt4 = [43,334]
        width_up = 512
        width_down = 140
        height = 189
    elif side == 'ront':
        pt1 = [170,230]
        pt2 = [467,230]
        pt3 = [587,441]
        pt4 = [46,441]
        width_up = 512
        width_down = 189
        height = 140
    elif side == 'back':
        pt1 = [173,230]
        pt2 = [466,230]
        pt3 = [571,427]
        pt4 = [67,427]
        width_up = 512
        width_down = 189
        height = 140
    elif side == 'ight':
        pt1 = [243,231]
        pt2 = [400,231]
        pt3 = [583,336]
        pt4 = [78,336]
        width_up = 512
        width_down = 140
        height = 189
    else:
        print('Error in directories!')
        return None
    
    pts_list = [pt1,pt2,pt3,pt4]
    # read input
    # specify desired output size 
    # specify conjugate x,y coordinates (not y,x)
    input_pts = np.float32(pts_list)
    #for val in input_pts:
    #    cv2.circle(img,(val[0],val[1]),5,(0,255,0),-1)
    output_pts = np.float32([[0,0], [width_up,0], [width_up - width_down,height], [width_down,height]])
    # compute perspective matrix
    matrix = cv2.getPerspectiveTransform(input_pts,output_pts)
    # do perspective transformation setting area outside input to black
    imgOutput = cv2.warpPerspective(img, matrix, (width_up,height), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    return imgOutput



    
def percpective_transform_no_files(undist_img,calibration_params,side):
    DIM = calibration_params[2]
    warped = warp(undist_img,DIM,side)
    return warped


def stitch(warped,car_symbol):
    left = warped[0]
    right = warped[1]
    front = warped[2]
    back = warped[3]
    front_ext = np.fliplr(np.vstack((front,np.zeros((372,512,3),np.uint8))))
    back_flipped = np.fliplr(np.rot90(back,k=2))
    back_ext = np.vstack((np.zeros((372,512,3),np.uint8),back_flipped))
    left_flipped = np.rot90(np.fliplr(left))
    left_ext = np.hstack((left_flipped,np.zeros((512,323,3),np.uint8)))
    right_flipped = np.rot90(np.fliplr(right),k=3)
    right_ext = np.hstack((np.zeros((512,323,3),np.uint8),right_flipped))
    
    front_ext[left_ext!=0] = 0
    front_ext[right_ext!=0] = 0
    back_ext[left_ext!=0] = 0
    back_ext[right_ext!=0] = 0
    
    left_ext[front_ext!=0] = 0
    left_ext[back_ext!=0] = 0
    right_ext[front_ext!=0] = 0
    right_ext[back_ext!=0] = 0
    
    #add car symbol
    car_symbol = cv2.resize(car_symbol,(134,232))
    top = 140
    bottom = top
    left = 189
    right = left
    borderType = cv2.BORDER_CONSTANT
    car_symbol = cv2.copyMakeBorder(car_symbol, top, bottom, left, right, borderType)
    
    
    bird_eye = front_ext + back_ext + left_ext + right_ext + car_symbol
    return bird_eye





def bird_eye_view(frames, calibration_params, car_symbol, img_dim = [640,480]):

    start = time.time()

    sizeimg = img_dim[0]*img_dim[1]*(3*1)

    imgL,imgR,imgT,imgB = frames
    
    calibration_params_left,calibration_params_right,calibration_params_front,calibration_params_back = calibration_params

    end = time.time()

    print(f'loading : {(end - start)}')

    start = time.time()

    undistL = undistort_fisheye_no_files(calibration_params_left,imgL,'left')
    undistR = undistort_fisheye_no_files(calibration_params_right,imgR,'right')
    undistF = undistort_fisheye_no_files(calibration_params_front,imgT,'front')
    undistB = undistort_fisheye_no_files(calibration_params_back,imgB,'back')

    end = time.time()

    print(f'undistort : {(end - start)}')

    start = time.time()

    warped = [0]*4

    warped[0] = percpective_transform_no_files(undistL,calibration_params_left,'left')
    warped[1] = percpective_transform_no_files(undistR,calibration_params_right,'ight')
    warped[2] = percpective_transform_no_files(undistF,calibration_params_front,'ront')
    warped[3] = percpective_transform_no_files(undistB,calibration_params_back,'back')

    end = time.time()

    print(f'warp : {(end - start)}')

    start = time.time()

    bird_view = stitch(warped,car_symbol)

    end = time.time()

    print(f'stitch : {(end - start)}')

    return bird_view



    


