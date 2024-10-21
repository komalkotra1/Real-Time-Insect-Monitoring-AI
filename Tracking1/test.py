import cv2
import numpy as np



# Polygon corner points coordinates
pts = np.array([[25, 70], [25, 160],
                [110, 200], [200, 160],
                [200, 70], [110, 20]],
               np.int32)

pts = pts.reshape((6, 1, 2))

print(pts)