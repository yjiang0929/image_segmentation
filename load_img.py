import cv2
from copy import deepcopy
import numpy as np
from argparse import ArgumentParser

from min_cut import Graph
# from create_graph import init_graph
from gaussian_graph import init_graph

class ImageLoader():
    def __init__(self):
        # initialize parser for arguments
        parser = self.build_parser()
        options = parser.parse_args()
        self.filename = options.filename
        self.img = []
        self.original = []

        # variables for labeling foreground and background
        self.fg_temp = np.empty((2,2), np.int32)
        self.bg_temp = np.empty((2,2), np.int32)
        self.fg_pixels = np.empty((0,3))
        self.bg_pixels = np.empty((0,3))
        self.mouse_down = False
        self.is_foreground = True

        self.run()

    def build_parser(self):
        # build parser for loading the filename
        parser = ArgumentParser()
        parser.add_argument('-f', type=str, dest='filename', help='filename', metavar='FILENAME', required=True)
        return parser

    def createLineIterator(self):
        """
        Modified Python line iterator implementation, credit: https://stackoverflow.com/questions/32328179/opencv-3-0-python-lineiterator
        Produces an array that consists of the RGB value of each pixel in a line between two points

        Parameters:
            -P1: a numpy array that consists of the coordinate of the first point (x,y)
            -P2: a numpy array that consists of the coordinate of the second point (x,y)
            -img: the image being processed

        Returns:
            -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])
        """
        img = self.img
        # choose the correct coordnates pair
        if self.is_foreground:
            P1 = self.fg_temp[0]
            P2 = self.fg_temp[1]
        else:
            P1 = self.bg_temp[0]
            P2 = self.bg_temp[1]

        #define local variables for readability
        imageH = img.shape[0]
        imageW = img.shape[1]
        P1X = P1[0]
        P1Y = P1[1]
        P2X = P2[0]
        P2Y = P2[1]

        #difference and absolute difference between points
        #used to calculate slope and relative location between points
        dX = P2X - P1X
        dY = P2Y - P1Y
        dXa = np.abs(dX)
        dYa = np.abs(dY)

        #predefine numpy array for output based on distance between points
        itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
        itbuffer.fill(np.nan)

        #Obtain coordinates along the line using a form of Bresenham's algorithm
        negY = P1Y > P2Y
        negX = P1X > P2X
        if P1X == P2X: #vertical line segment
           itbuffer[:,0] = P1X
           if negY:
               itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
           else:
               itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
        elif P1Y == P2Y: #horizontal line segment
           itbuffer[:,1] = P1Y
           if negX:
               itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
           else:
               itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
        else: #diagonal line segment
           steepSlope = dYa > dXa
           if steepSlope:
               slope = dX.astype(np.float32)/dY.astype(np.float32)
               if negY:
                   itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
               else:
                   itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
               itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int) + P1X
           else:
               slope = dY.astype(np.float32)/dX.astype(np.float32)
               if negX:
                   itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
               else:
                   itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
               itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int) + P1Y

        #Remove points outside of image
        colX = itbuffer[:,0]
        colY = itbuffer[:,1]
        itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

        #Get RGB intensities from img ndarray
        return img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]

    def on_mouse(self, event, x, y, flags, params):
        # register mouse up/down and record a line
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_down = True
            if self.is_foreground:
                self.fg_temp[0] = (x,y)
            else:
                self.bg_temp[0] = (x,y)

        if event == cv2.EVENT_LBUTTONUP and self.mouse_down:
            # record an event when the mouse is raised after press
            self.mouse_down = False
            if self.is_foreground:
                self.fg_temp[1] = (x,y)
                # save pixels on the line
                if len(self.fg_pixels)>0:
                    self.fg_pixels = np.append(self.fg_pixels, self.createLineIterator(), axis=0)
                else:
                    self.fg_pixels = np.array(self.createLineIterator())

                # draw the lines on the image
                cv2.line(self.img, (self.fg_temp[0][0],self.fg_temp[0][1]),
                         (self.fg_temp[1][0],self.fg_temp[1][1]),(255,0,0), 2)
            else:
                self.bg_temp[1] = (x,y)
                # save pixels on the line
                if len(self.bg_pixels)>0:
                    self.bg_pixels = np.append(self.bg_pixels, self.createLineIterator(), axis=0)
                else:
                    self.bg_pixels = np.array(self.createLineIterator())

                # draw the lines on the image
                cv2.line(self.img, (self.bg_temp[0][0],self.bg_temp[0][1]),
                         (self.bg_temp[1][0],self.bg_temp[1][1]),(0,255,0), 2)

    def load_img(self):
        # self.img = cv2.imread(self.filename,0)
        self.img = cv2.imread(self.filename)
        self.original = deepcopy(self.img)
        cv2.namedWindow('image')
        cv2.setMouseCallback("image", self.on_mouse, 0)
        # Update the display with lines
        while True:
            cv2.imshow('image', self.img)
            k = cv2.waitKey(5) & 0xFF

            # switch between drawing foreground and background
            if k == ord('m'):
                print('switch mode')
                self.is_foreground = not self.is_foreground
            elif k==27:
                break

        cv2.destroyAllWindows()

        print('Number of foreground pixels: ', np.shape(self.fg_pixels))
        print('Number of background pixels: ', np.shape(self.bg_pixels))

    def run(self):
        self.load_img()

        # run image segmentation
        k = 1
        sigma = 100
        segments = init_graph(self.original, k, sigma, self.fg_pixels, self.bg_pixels)
        print("Image Segemented")
        # generate foreground mask
        m, n = self.img.shape[0], self.img.shape[1]
        back_mask = np.zeros((m,n), np.uint8)
        for i,p in enumerate(segments):
            if not p and (i<(m*n)):
                row = i // n
                col = i % n
                back_mask[row][col] = 1

        fore_mask = np.zeros((m,n), np.uint8)
        for i,p in enumerate(segments):
            if p and (i<(m*n)):
                row = i // n
                col = i % n
                back_mask[row][col] = 1

        fore_img = cv2.bitwise_and(self.original, self.original, mask=fore_mask)
        back_img = cv2.bitwise_and(self.original, self.original, mask=back_mask)
        while 1:
            cv2.imshow("fore", fore_img)
            cv2.imshow("back", back_img)
            k = cv2.waitKey(5) & 0xFF
            if k==27:
                break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    imgLoader = ImageLoader()
