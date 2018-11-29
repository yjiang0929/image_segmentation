import cv2
import numpy as np
from argparse import ArgumentParser

class ImageLoader():
    def __init__(self):
        parser = self.build_parser()
        options = parser.parse_args()
        self.filename = options.filename
        self.img = []

        # variables for labeling foreground and background
        self.fg_temp = np.empty((2,2), np.uint32)
        self.bg_temp = np.empty((2,2), np.uint32)
        self.mouse_down = False
        self.fg_mode = True

        self.load_img()

    def build_parser(self):
        parser = ArgumentParser()
        parser.add_argument('-f', type=str, dest='filename', help='filename', metavar='FILENAME', required=True)
        return parser

    def on_mouse(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_down = True
            if self.fg_mode:
                self.fg_temp[0] = (x,y)
            else:
                self.bg_temp[0] = (x,y)

        if event == cv2.EVENT_LBUTTONUP and self.mouse_down:
            self.mouse_down = False
            if self.fg_mode:
                self.fg_temp[1] = (x,y)
                cv2.line(self.img, (self.fg_temp[0][0],self.fg_temp[0][1]),
                         (self.fg_temp[1][0],self.fg_temp[1][1]),(255,0,0), 2)
            else:
                self.bg_temp[1] = (x,y)
                cv2.line(self.img, (self.bg_temp[0][0],self.bg_temp[0][1]),
                         (self.bg_temp[1][0],self.bg_temp[1][1]),(0,255,0), 2)

    def load_img(self):
        self.img = cv2.imread(self.filename)
        cv2.namedWindow('image')
        cv2.setMouseCallback("image", self.on_mouse, 0)
        while True:
            cv2.imshow('image', self.img)
            k = cv2.waitKey(5) & 0xFF
            if k == ord('m'):
                print('switch mode')
                self.fg_mode = not self.fg_mode
            elif k==27:
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    imgLoader = ImageLoader()
