from tkinter import E
import cv2
import chess.svg
from chessboard import display
from ChessPiece import *
import numpy as np
from matplotlib import pyplot as plt
import json
from time import time, sleep
import os
import glob
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import copy
import argparse
from getch import getch
import mediapipe as mp

whT = 320
conf_threshold = 0.2
nms_threshold = 0.5

# Pauses the game on spacebar button press
def pause_game(pause_flag, pause_ctr):
    while True:
        try: 
            key = getch()
            if ord(key) == 32:
                pause_ctr.value = (pause_ctr.value+1)%2
                if pause_ctr.value == 1:
                    print("\tGame paused")
                    pause_flag.value = 1
                else:
                    print("\tGame resumed")
                    pause_flag.value = 0   
        except KeyboardInterrupt:
            print("\n\tGame interrupted!")

def get_chess_tiles(img, show = 0):
    if show == 1:
        # Display original image
        cv2.imshow('Original', img)
        cv2.waitKey(0)

    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
    if show == 1:
        cv2.imshow('Blurred and gray-scaled', img_blur)
        cv2.waitKey(0)

    # Sobel Edge Detection
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
    # Display Sobel Edge Detection Images
    if show == 1:
        cv2.imshow('Sobel X', sobelx)
        cv2.waitKey(0)
        cv2.imshow('Sobel Y', sobely)
        cv2.waitKey(0)
        cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
        cv2.waitKey(0)

    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
    # Display Canny Edge Detection Image
    if show == 1:
        cv2.imshow('Canny Edge Detection', edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return (sobelx, sobely, sobelxy, edges)

def get_frame_with_chess_pieces_located_images(img, chess_pieces):
    chess_pieces_located_images = list()
    frame_with_chess_pieces_located_images =  np.zeros(img.shape)
    # Get all individual masks of small image boxes and save them into a single image
    for el in chess_pieces:
        frame_with_chess_pieces_located_images += el.get_ChessPiece_located_image(img)

    return frame_with_chess_pieces_located_images

# initializes yolo darknet model ---> returns yolo darknet model, class names
def yolo_init():
    # Opening yolo_init.json file
    with open('yolo_init.json') as yolo_init_json_file:
        yolo_init_dict = json.load(yolo_init_json_file)

    # set custom classes file
    with open(yolo_init_dict["classesFile"],'rt') as f:
        className = f.read().rstrip('\n').split('\n')

    # set custom model configuration
    modelConfiguration = yolo_init_dict["modelConfiguration"]

    # set custom model weights
    modelWeights = yolo_init_dict["modelWeights"]

    # set YOLO-v3 network
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    return net, className

# finds chess pieces with respective bounding boxes ---> returns array of chessPiece objects
def get_chess_pieces(outputs, img, chess_board_tiles_dict, chess_board_square, className, frame_id = 0, save_images = 1, show = 0):
    hT, wT, cT = img.shape
    bbox, confs, classIDs = [], [], []
    chess_pieces = list()

    # Get all chess pieces and respective bounding boxes
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > conf_threshold:
                w, h = int(detection[2]*wT), int(detection[3]*wT)
                x, y = int(detection[0]*wT - w/2), int(detection[1]*hT - h/2)
                if chess_board_square.contains(Point(x+w/2, y+h/2)):
                    bbox.append([x, y, w, h])
                    classIDs.append(classID)
                    confs.append(float(confidence))

    # Delete close duplicate bounding boxes
    indices = cv2.dnn.NMSBoxes(bbox, confs, conf_threshold, nms_threshold)

    # Append the correct remaining chess pieces in the list
    if show == 1:
        print("\tPrinting chess pieces...")
    for counter, object_i in enumerate(indices):
        i = int(object_i)
        box = bbox[i]
        # Draw bounding boxes and save the new image
        if save_images == 1:
            xd, yd, wd, hd = box[0], box[1], box[2], box[3]
            cv2.rectangle(img, (xd, yd), (xd + wd, yd + hd), (0, 255, 0), 2)
            cv2.putText(img, f'{className[classIDs[i]]}{int(confs[i]*100)}%',
                    (xd, yd-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

        # Fetch the data and use the to initialize a temporary buffer object
        x, y, w, h = box[0]+box[2]/2, box[1]+box[3]/2, box[2], box[3]
        temp_chessPiece = ChessPiece(label = classIDs[i], x = x, y = y, width = w, height = h, confidence = confs[i])
        temp_chessPiece.set_currPosition(chess_board_tiles_dict)
        if show == 1:
            print(counter+1, temp_chessPiece)
        chess_pieces.append(temp_chessPiece)

    if save_images == 1:
        cv2.putText(img, str(frame_id), (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite("output_images/Chess_pieces_bounding_boxes.jpg", img)
        cv2.imwrite("output_images/Current_frame.jpg", img)
    if show == 1:
        cv2.imshow('Chess pieces bounding boxes', img)
        cv2.waitKey(0)
        print("\tThis frame contains", len(chess_pieces), "chess pieces.")

    return chess_pieces

# initializes the chess_board_tiles_dict
def chess_board_tiles_init():
    # Create dict with tile names as keys
    chess_board_tiles_dict = {"A8": -1,"B8": -1,"C8": -1,"D8": -1,"E8": -1,"F8": -1,"G8": -1,"H8": -1,
                            "A7": -1,"B7": -1,"C7": -1,"D7": -1,"E7": -1,"F7": -1,"G7": -1,"H7": -1,
                            "A6": -1,"B6": -1,"C6": -1,"D6": -1,"E6": -1,"F6": -1,"G6": -1,"H6": -1,
                            "A5": -1,"B5": -1,"C5": -1,"D5": -1,"E5": -1,"F5": -1,"G5": -1,"H5": -1,
                            "A4": -1,"B4": -1,"C4": -1,"D4": -1,"E4": -1,"F4": -1,"G4": -1,"H4": -1,
                            "A3": -1,"B3": -1,"C3": -1,"D3": -1,"E3": -1,"F3": -1,"G3": -1,"H3": -1,
                            "A2": -1,"B2": -1,"C2": -1,"D2": -1,"E2": -1,"F2": -1,"G2": -1,"H2": -1,
                            "A1": -1,"B1": -1,"C1": -1,"D1": -1,"E1": -1,"F1": -1,"G1": -1,"H1": -1
                            }
    return chess_board_tiles_dict

# updates the chess_board_tiles_dict
def set_chess_board_tiles(img, chess_board_tiles_dict, draw_corners = 0, save_images = 1, show =0):
    # Defining the dimensions of checkerboard
    CHECKERBOARD = (7,7)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray[gray>180] = 255
    gray[gray<100] = 0
    # Find the chess board corners
    ret, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_ACCURACY+ cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
    if ret == True:
        corners1 = copy.copy(corners)
        corners1 = corners1.reshape((7,7,1,2))
        corners2 = np.zeros((9,9,1,2))
        dx = corners1[:,1,:,0]-corners1[:,0,:,0]
        dy = corners1[1,:,:,1]-corners1[0,:,:,1]
        corners2[1:-1,1:-1,:,:] = corners1[:,:,:,:]
        corners2[0,1:-1,:,0] = corners1[0,:,:,0]
        corners2[0,1:-1,:,1] = corners1[0,:,:,1]-dy
        corners2[-1,1:-1,:,0] = corners1[-1,:,:,0]
        corners2[-1,1:-1,:,1] = corners1[-1,:,:,1]+dy
        corners2[:,0,:,1] = corners2[:,1,:,1]
        corners2[:,0,:,0] = corners2[:,1,:,0]- dx[0]
        corners2[:,-1,:,1] = corners2[:,-2,:,1]
        corners2[:,-1,:,0] = corners2[:,-2,:,0]+dx[0]
        corners3 = np.zeros((81,1,2))
        j = 0
        for row in corners2:
            for col in row:
                corners3[j,:,:] = col
                j+=1
        # Fill the dict with quadruples of points to define bounding boxes
        j = 0
        for id, key in enumerate(chess_board_tiles_dict.keys()):
            points =np.array([[corners3[id+j,0,0], corners3[id+j,0,1]],
                            [corners3[id+j+1,0,0], corners3[id+j+1,0,1]],
                            [corners3[id+j+10,0,0], corners3[id+j+10,0,1]],
                            [corners3[id+j+9,0,0], corners3[id+j+9,0,1]]], dtype = np.int32)
            chess_board_tiles_dict[key] = Polygon(points)
            if draw_corners == 1:
                cv2.polylines(img, [points], 1, (0,0,255),2)
                cv2.putText(img, str(key),(int(corners3[id+j+9,0,0])+10, int(corners3[id+j+9,0,1])-18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if id%8 == 7:
                j+=1
        # Draw and display the corners
        if draw_corners == 1:
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        if save_images == 1:
            cv2.imwrite("output_images/Chess_board_tiles_bounding_boxes.jpg",img)
        if show == 1:
            cv2.imshow('Chess board tiles bounding boxes', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    return ret

def get_chess_board(chess_pieces):
    chess_board = [['*','*','*','*','*','*','*','*'], # 8
                   ['*','*','*','*','*','*','*','*'], # 7
                   ['*','*','*','*','*','*','*','*'], # 6
                   ['*','*','*','*','*','*','*','*'], # 5
                   ['*','*','*','*','*','*','*','*'], # 4
                   ['*','*','*','*','*','*','*','*'], # 3
                   ['*','*','*','*','*','*','*','*'], # 2
                   ['*','*','*','*','*','*','*','*']] # 1
                   # A # B # C # D # E # F # G # H
    for chess_piece in chess_pieces:
        pos = 8-int(chess_piece.currPosition[1]), ord(chess_piece.currPosition[0])-65
        chess_board[pos[0]][pos[1]] = chess_piece.piece.symbol()
    blanks_counter = 0
    chess_board_string = ""
    for row in chess_board:
        blanks_counter = 0
        for el in row:
            if el == '*':
                blanks_counter+=1
            else:
                if blanks_counter!=0:
                    chess_board_string = chess_board_string + str(blanks_counter) + el
                    blanks_counter = 0
                else:
                    chess_board_string = chess_board_string + el
                    blanks_counter = 0
        if blanks_counter != 0:
            chess_board_string = chess_board_string + str(blanks_counter)
        chess_board_string+='/'
    chess_board_string = chess_board_string[:-1] +  " w - - 0 1"
    return chess_board_string
    # arrows=[chess.svg.Arrow(chess.E4, chess.F6, color="#0000cccc")],squares=chess.SquareSet(chess.BB_DARK_SQUARES & chess.BB_FILE_B)

def show_chess_board(chess_board_string):
    game = display.start(chess_board_string,(0,200,255), 'Grand Master Chess')
    # Checking GUI window for QUIT event. (Esc or GUI CANCEL)
    try:
        while True:
            display.check_for_quit()
    except:
        try:
            display.terminate()
        except:
            print("Error hase occurred during game. Leaving the game...")
    finally:
        display.terminate()
        print("Game was closed! Game data are lost!")

# Vizualize chess game
def update_chess_board(chess_board_string, frame_id):
    game = display.start(chess_board_string,(0,200,255), 'Grand Master Chess \nTurn '+str(frame_id))
    # Checking GUI window for QUIT event. (Esc or GUI CANCEL)
    display.check_for_quit()
    # display.update(chess_board_string, game)
