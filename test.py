
import numpy as np
import cv2, imutils
import glob


images = glob.glob('Calib_img/*.bmp')
images = glob.glob('Calib_img/*.bmp')
i = 0
for image in images:

    img = cv2.imread(image)
    img = imutils.resize(img, width=640)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners


    # If found, add object points, image points (after refining them)

    cv2.imwrite('Calib_img/' + str(i)+ '.png',gray)
    i += 1
        # Draw and display the corners
    cv2.imshow('img', img)
    cv2.waitKey(500)


cv2.destroyAllWindows()

