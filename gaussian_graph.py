from numpy import *
from sklearn import mixture
import maxflow

# use RGB color
def init_graph(img_array, k, sigma, F_Pixels, B_Pixels):
    # k = similar pixels have weight close to kappa
    # Sigma value --> determines how fast the values decay towards zero with increasing dissimilarity.
    I, F_Pixels, B_Pixels = array(img_array), array(F_Pixels), array(B_Pixels) # convert images to numpy arrays
    x_train = concatenate([F_Pixels, B_Pixels]) # stack input to training data, foreground and background
    clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
    clf.fit(x_train) # fit GMM model

    Im = I.reshape(-1,3) #converting the image array to a vector for ease
    m,n = I.shape[0],I.shape[1]# copy the size

    g = maxflow.Graph[int](m, n)
    nodes = g.add_nodes(m*n)
    # compute likelihoods
    for i in range(0, n*m):#checking the 4-neighborhood pixels
        weights = clf.predict_proba([Im[i]])
        source_weight = weights[0][0]
        sink_weight = weights[0][1]
        g.add_tedge(i, source_weight, sink_weight)

        # four neighboring pixels
        if i%n != 0: # for left pixels
            w = k*exp(-(linalg.norm(Im[i]-Im[i-1]))/sigma) # the cost function for two pixels is the frobenous norm between pixels
            g.add_edge(i, i-1, w, k-w)

        if (i+1)%n != 0: # for right pixels
            w = k*exp(-(linalg.norm(Im[i]-Im[i+1]))/sigma)
            g.add_edge(i, i+1, w, k-w)

        if i//n != 0: # for top pixels
            w = k*exp(-(linalg.norm(Im[i]-Im[i-n]))/sigma)
            g.add_edge(i, i-n, w, k-w)

        if i//n != m-1: # for bottom pixels
            w = k*exp(-(linalg.norm(Im[i]-Im[i+n]))/sigma)
            g.add_edge(i, i+n, w, k-w)

    # compute max flow
    g.maxflow()
    Iout = ones(shape = nodes.shape)
    for i in range(len(nodes)):
        Iout[i] = g.get_segment(nodes[i])

    return Iout
