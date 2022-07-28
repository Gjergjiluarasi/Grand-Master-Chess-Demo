from numpy import double
import numpy as np
import cv2
import json
from shapely.geometry import Point
from chess import Piece
from chess import PieceType
from chess import Color

WIDTH = 640
HEIGHT = 480

class ChessPiece:
    def __init__(self, label=None, x=None, y=None, width=None, height=None, confidence = None, prevPosition = '', currPosition = ''):
        self.label = label
        self.x = x
        self.y = y
        self.center = Point(self.x, self.y)
        self.width = width
        self.height = height
        self.confidence = confidence
        self.prevPosition = prevPosition
        self.currPosition = currPosition
        self.piece = Piece(self.label%6+1,  self.label<6)

    def __str__(self):
        # Opening yolo_init.json file
        with open('yolo_init.json') as yolo_init_json_file:
            yolo_init_dict = json.load(yolo_init_json_file)
        # Getting class names
        with open(yolo_init_dict["classesFile"],'rt') as f:
            className = f.read().rstrip('\n').split('\n')
        rep = 'ChessPiece(label=' + className[self.label] + ', x=' + str(self.x) + ', y=' + str(self.y) + \
            ', width='+  str(self.width) +  ', height=' + str(self.height) + \
            ', confidence=' + str(self.confidence) + ', prevPosition=' + self.prevPosition + \
            ', currPosition=' + self.currPosition + ')'
        return rep

    def get_ChessPiece_located_image(self, src_img):
        img = np.zeros(src_img.shape)
        x_min = int(self.y - self.height/2)
        x_max = int(self.y + self.height/2) 
        y_min = int(self.x - self.width/2)
        y_max = int(self.x + self.width/2)
        img[x_min:x_max, y_min:y_max, :] = src_img[x_min:x_max, y_min:y_max, :]
        return img

    def show_ChessPiece_located_image(self, src_img):
        img = np.zeros(src_img.shape)
        x_min = int(self.y - self.height/2)
        x_max = int(self.y + self.height/2) 
        y_min = int(self.x - self.width/2)
        y_max = int(self.x + self.width/2)
        img[x_min:x_max, y_min:y_max, :] = src_img[x_min:x_max, y_min:y_max, :]
        cv2.imwrite('tmp.jpg', img)
        cv2.imshow('ChessPiece located image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 

    def set_prevPosition(self, val):
        self.prevPosition = val
        
    def set_currPosition(self, chess_board_tiles_dict):
        for tile in chess_board_tiles_dict.keys():
            if chess_board_tiles_dict[tile].contains(self.center):
                self.currPosition = tile
                break
        if self.currPosition != '':
            return True
        else:
            return False 