from numpy import *
import numpy
from PIL import Image
import cv2

def init_graph(file, kappa, sigma, F_Pixels, B_Pixels):
    I = (Image.open(file).convert('L')) # read image, convert to greyscale
    I, F_Pixels, B_Pixels = array(I), array(F_Pixels), array(B_Pixels) # convert images to numpy arrays
    #Taking the mean of the histogram
    Ibmean = mean(cv2.calcHist([Ib],[0],None,[256],[0,256]))
    Ifmean = mean(cv2.calcHist([If],[0],None,[256],[0,256]))
    #initalizing the foreground/background probability vector
    F = ones(shape = I.shape)
    B = ones(shape = I.shape)
    Im = I.reshape(-1,1) #converting the image array to a vector for ease
    m,n = I.shape[0],I.shape[1]# copy the size
    for i in range(I.shape[0]): # Defining the Probability function....
        for j in range(I.shape[1]):
            F[i,j] = -log(abs(I[i,j] - Ifmean)/(abs(I[i,j] - Ifmean)+abs(I[i,j] - Ibmean))) # Probability of a pixel being foreground
            B[i,j] = -log(abs(I[i,j] - Ibmean)/(abs(I[i,j] - Ibmean)+abs(I[i,j] - Ifmean))) # Probability of a pixel being background
    F,B = F.reshape(-1,1),B.reshape(-1,1) # convertingb  to column vector for ease
    for i in range(Im.shape[0]):
        Im[i] = Im[i] / linalg.norm(Im[i]) # normalizing the input image vector

    g = np.array([[inf, 0, 0],
                  [inf, 0, 0],
                  [inf, 0, 0]
                 ])
    source = 0
    sink = m*n + 1

    for i in range(m*n):#checking the 4-neighborhood pixels
        ws=(F[i]/(F[i]+B[i])) # source weight
        wt=(B[i]/(F[i]+B[i])) # sink weight
        g.add_tedge(i,ws[0],wt) # edges between pixels and terminal
        if i%n != 0: # for left pixels
            w = k*exp(-(abs(Im[i]-Im[i-1])**2)/s) # the cost function for two pixels
            g.add_edge(i,i-1,w[0],k-w[0]) # edges between two pixels
            '''Explaination of the likelihood function: * used Bayes’ theorem for conditional probabilities
            * The function is constructed by multiplying the individual conditional probabilities of a pixel being either
            foreground or background in order to get the total probability. Then the class with highest probability is selected.
            * for a pixel i in the image:
                               * weight from sink to i:
                               probabilty of i being background/sum of probabilities
                               * weight from source to i:
                               probabilty of i being foreground/sum of probabilities
                               * weight from i to a 4-neighbourhood pixel:
                                K * e−|Ii−Ij |2 / s
                                 where k and s are parameters that determine hwo close the neighboring pixels are how fast the values
                                 decay towards zero with increasing dissimilarity
            '''
        if (i+1)%n != 0: # for right pixels
            w = k*exp(-(abs(Im[i]-Im[i+1])**2)/s)
            g.add_edge(i,i+1,w[0],k-w[0]) # edges between two pixels
        if i//n != 0: # for top pixels
            w = k*exp(-(abs(Im[i]-Im[i-n])**2)/s)
            g.add_edge(i,i-n,w[0],k-w[0]) # edges between two pixels
        if i//n != m-1: # for bottom pixels
            w = k*exp(-(abs(Im[i]-Im[i+n])**2)/s)
            g.add_edge(i,i+n,w[0],k-w[0]) # edges between two pixels


    ## graph is 2d list of lists, not np array
    return g.tolist()
