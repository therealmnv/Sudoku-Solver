import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from KNN import *
from datasetGeneration import *

solvedArray = np.zeros((9,9),np.int8)


def isValid(x, y, val,grid):
    for i in range(9):
        if grid[x][i] == val:
            return False

    for i in range(9):
        if grid[i][y] == val:
            return False

    x_new = (x // 3) * 3
    y_new = (y // 3) * 3

    for i in range(3):
        for j in range(3):
            if grid[x_new + i][y_new + j] == val:
                return False

    return True

def solve(grid):
    for i in range(9):
        for j in range(9):
            if grid[i][j] == 0:
                for v in range(1, 10):
                    if isValid(i, j, v,grid):
                        grid[i][j] = v
                        solve(grid)
                        grid[i][j] = 0
                return 

    global solvedArray

    for i in range(9):
        for j in range(9):
            solvedArray[i][j] = grid[i][j]


def img2array(img):
    rows = img.shape[0]
    cols = img.shape[1]
    img_size = rows*cols
    return img.reshape(img_size)


def largestSquare(img,size,imgOg):
    th = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,23,9)
    median = cv.medianBlur(th,3)

    contours,_ = cv.findContours(median, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    maxArea = cv.contourArea(contours[0])
    cnt = 0
    for i in contours:
        if cv.contourArea(i)>maxArea:
            cnt = i
            maxArea = cv.contourArea(i)

    #CREATING A MASK TO FIND THE CORNERS WITH cv.goodFeaturesToTrack
    blank = np.zeros(img.shape, np.uint8)
    cv.fillPoly(blank, [cnt],color=(255,255,255))
    blank = np.float32(blank)

    # detect corners with the goodFeaturesToTrack function.
    corners = cv.goodFeaturesToTrack(blank, 4, 0.01, 150)
    corners = np.int0(corners)

    # we iterate through each corner, 
    # making a circle at each point that we think is a corner.
    for i in corners:
        x, y = i.ravel()
        cv.circle(imgOg, (x, y), 3, 255, -1)


    #GET corners IN DESIRED SHAPE FOR cv.getPerspectiveTransform
    corners = np.float32(corners)

    pts1 = []
    for i in range(4):
        curr=[]
        curr.append(corners[i][0][0])
        curr.append(corners[i][0][1])
        pts1.append(curr)

    for i in range(4):
        if pts1[i][0] < size/2 and pts1[i][1] < size/2: #top left
            temp = pts1[0]
            pts1[0] = pts1[i]
            pts1[i] = temp

    for i in range(3):
        if pts1[i+1][0] > size/2 and pts1[i+1][1] < size/2: #top right
            temp = pts1[0]
            pts1[0] = pts1[i]
            pts1[i] = temp

    for i in range(2):
        if pts1[i+2][0] < size/2 and pts1[i+2][1] > size/2: #bottom left
            temp = pts1[0]
            pts1[0] = pts1[i]
            pts1[i] = temp

    pts1 = np.float32(pts1)
    pts2 = np.float32([[0,0],[size,0],[0,size],[size,size]])

    M = cv.getPerspectiveTransform(pts1,pts2)
    imgWarped = cv.warpPerspective(imgOg,M,(size,size))
    return imgWarped, pts1, pts2

def extractNumbers(imgWarpedGray,size):
    th = cv.adaptiveThreshold(imgWarpedGray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,39,4)
    th2 = th.copy()

    cnts = cv.findContours(th2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1] 

    #REMOVING NOISE
    for i in cnts:
        area = cv.contourArea(i)
        if area < size*2.5: #1000
            cv.drawContours(th2, [i], -1, (0,0,0), -1)

    th2 = cv.bitwise_not(th2)
    uncleanNums = cv.bitwise_and(th,th,mask=th2)
    
    cnts = cv.findContours(uncleanNums, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1] 

    for i in cnts:
        area = cv.contourArea(i)
        if area < size/4: #100
            cv.drawContours(uncleanNums, [i], -1, (0,0,0), -1)
    
    # cleanNums = uncleanNums.copy()

    # init = size//9 
    # for i in range(8): 
    #     cv.line(cleanNums,(init,0),(init,size),(255,255,255),1) 
    #     init+=size//9 
 
    # init = size//9 
    # for i in range(8):
    #     cv.line(cleanNums,(0,init),(size,init),(255,255,255),1)
    #     init+=size//9

    a = np.empty((9,9),int)
    y = 0
    
    listMatrix = []
    cellSizeY = size//9
    for i in range(9):
        cellSizeX = size//9
        x = 0
        row = []
        for j in range(9):
            element = uncleanNums[y:cellSizeY,x:cellSizeX]
            element = cv.resize(element,(28,28),interpolation = cv.INTER_AREA)
            eleVec = image2array(element)
            eleVec = np.array(eleVec)
            eleVec = np.divide(eleVec,255,dtype='float32')
            num = predictor(eleVec.reshape(1,-1))
            row.append(int(num[0]))
            cellSizeX += size//9
            x += size//9
        listMatrix.append(row)
        cellSizeY += size//9
        y += size//9 
    listArray = np.array(listMatrix)
    
    return listArray, uncleanNums

def main():
    #TAKING INPUT AND CONVERTING TO GRAYSCALE
    imgOg = cv.imread("sudoku.jpg")
    size = imgOg.shape[0]
    
    imGray = cv.cvtColor(imgOg,cv.COLOR_BGR2GRAY)
    if imGray is None:
        sys.exit("Could not read the image.")

    #FINDING THE LARGEST SQUARE (i.e. THE 9*9 SUDOKU)
    imgWarped, pts1, pts2 = largestSquare(imGray,size,imgOg)

    imgWarpedGray = cv.cvtColor(imgWarped,cv.COLOR_BGR2GRAY)

    listArray, uncleanNums = extractNumbers(imgWarpedGray,size)
    
    solve(listArray)
    
    blank = np.zeros((size,size), np.uint8)

    folderpath = 'D:\Java Programs (VS Studio)\Python\Sudoku Solver\dataset/nums'

    cellSizeY = size//9
    y=0
    for i in range(9):
        cellSizeX = size//9
        x = 0
        for j in range(9):
            if listArray[i][j] == 0:            
                path = os.path.join(folderpath,str(solvedArray[i][j])+'.jpeg')

                img = cv.imread(path,0)
                
                _,thresh1 = cv.threshold(img,30,255,cv.THRESH_BINARY)

                element = cv.resize(thresh1,(size//9,size//9),interpolation = cv.INTER_AREA)
                
                blank[y:cellSizeY,x:cellSizeX] = element[0:size//9,0:size//9]

            cellSizeX += size//9
            x += size//9

        cellSizeY += size//9
        y += size//9
    
    M = cv.getPerspectiveTransform(pts2,pts1)
    
    blank = cv.warpPerspective(blank,M,(imGray.shape[1],imGray.shape[0]))
    blank = cv.bitwise_not(blank)

    blank = cv.bitwise_and(blank,imGray)

    cv.imshow('frame',blank)
    k=cv.waitKey(0)

if __name__=="__main__":
    main()