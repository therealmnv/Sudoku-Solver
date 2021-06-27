# Sudoku-Solver
-main.py takes in the sudoku image (sudoku.jpg) and preprocesses it.  
-Preprocessing of the image is done with OpenCV:  
  \*finding the largest square (in most cases the sudoku 9*9 board),  
  \*applying warp transformation to the largest square found,  
  \*extracting digits by segmenting the image  
-The extracted image of a particular digit is predicted to be a number between 0-9 using KNN (0 represents a blank position to be filled).  
  \*KNN (KNN.py) is trained on arrays of TEXT DIGITS dataset found on Kaggle by Kshitij Dhama: https://www.kaggle.com/kshitijdhama/printed-digits-dataset  
  \*since the dataset is really small, I augmented the images to get 1000 images for each class (imageAugmentation.py)  
  \*datasetGeneration.py converts all the images to array into a .csv file  
-Once we correctly detect the digits, we send the 2D numpy array to the recursive solve() function which uses backtracking to solve the sudoku.   
-After the sudoku is solved we focus on projecting the solved image of digits to the original sudoku and we're done!
