import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2 as cv

def image2array(img):
    if len(img.shape) == 3:
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        
    rows,cols = img.shape
    img_size = rows*cols
    return img.reshape(img_size)
# Parent Directory path
parent_dir = "D:/Java Programs (VS Studio)/Python/Sudoku Solver/dataset/"

l = []
y = []

for i in os.listdir(parent_dir):
    path = os.path.join(parent_dir, i)

    for filename in os.listdir(path): 
        filepath = os.path.join(path, filename)

        img = cv.imread(filepath,0)
        _,thresh1 = cv.threshold(img,50,255,cv.THRESH_BINARY)
        
        img_1D_vector = image2array(thresh1)     
        l.append(img_1D_vector)
        y.append(i)
    
cols = list(range(0,len(l[0])))

X = np.array(l)
y = np.array(y)

df = pd.DataFrame(data = X, columns = cols)
df2 = pd.DataFrame(data = y, columns = ['number'])

df.to_csv("X.csv")
df2.to_csv("y.csv")
