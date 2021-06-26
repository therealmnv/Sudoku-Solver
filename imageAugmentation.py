# Importing necessary library
import Augmentor
import os
# Passing the path of the image directory
parent_dir = "D:\Java Programs (VS Studio)\Python\Sudoku Solver/0"

for i in os.listdir(parent_dir):
    path = os.path.join(parent_dir, i) 
    p = Augmentor.Pipeline(path)
    p.flip_left_right(0.2)
    p.rotate(0.4, 15, 15)
    p.skew(0.3, 0.5)
    p.zoom(probability = 0.3, min_factor = 0.8, max_factor = 1.3)
    p.sample(1000)