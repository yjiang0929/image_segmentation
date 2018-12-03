from numpy import *
import numpy
from PIL import Image
import cv2

def create_graph(file, kappa, sigma, F_Pixels, B_Pixels):
    I = (Image.open(file).convert('L')) # read image, convert to greyscale
    I, F_Pixels, B_Pixels = array(I), array(F_Pixels), array(B_Pixels) # convert images to numpy arrays
    #Taking the mean of the histogram
    Ibmean = mean(cv2.calcHist([Ib],[0],None,[256],[0,256]))
    Ifmean = mean(cv2.calcHist([If],[0],None,[256],[0,256]))
    #initalizing the foreground/background probability vector
    F = ones(shape = I.shape)
    B = ones(shape = I.shape)
    
