import decimal

import numpy as np
import cv2 as cv
import glob
import copy
import sys
import math
import os
import pathlib
from decimal import Decimal
from PyQt5.QtWidgets import QFileDialog

pixelToWorldMatrix = np.identity(3)
def draw_crosshair(image, center, width, color):
    cv.line(image, (center[0] - width // 2, center[1]), (center[0] + width // 2, center[1]), color, 9)
    cv.line(image, (center[0], center[1] - width // 2), (center[0], center[1] + width // 2), color, 9)


# Used to convert 1x3 rodrigues rotation matrix to 3x3 rotation matrix
def rodrigues_vec_to_rotation_mat(rodrigues_vec):
    theta = np.linalg.norm(rodrigues_vec)
    if theta < sys.float_info.epsilon:
        rotation_mat = np.eye(3, dtype=float)
    else:
        r = rodrigues_vec / theta
        I = np.eye(3, dtype=float)
        r_rT = np.array([
            [r[0]*r[0], r[0]*r[1], r[0]*r[2]],
            [r[1]*r[0], r[1]*r[1], r[1]*r[2]],
            [r[2]*r[0], r[2]*r[1], r[2]*r[2]]
        ])
        r_cross = np.array([
            [0, -r[2], r[1]],
            [r[2], 0, -r[0]],
            [-r[1], r[0], 0]
        ])
        rotation_mat = math.cos(theta) * I + (1 - math.cos(theta)) * r_rT + math.sin(theta) * r_cross
    return rotation_mat

def loadCalibration():
    path = './calibSaves'
    isExist = os.path.exists(path)
    if isExist:
        pass
    else:
        print('[WARNING] calibSaves folder not found.')
        print('\nMake sure to run the calibration first.')
        return 0
    filename = QFileDialog.getOpenFileName(directory='./calibSaves')[0]
    rootpath = pathlib.Path().resolve()
    removepath = str(rootpath) + '\calibSaves'
    global pixelToWorldMatrix
    pixelToWorldMatrix = np.load(filename)
    print('\nMatrix loaded')
    print(pixelToWorldMatrix, '\n')
    return os.path.relpath(filename, removepath)

# chessboardSize, frameSize, chessboardSquareMM
def runCalibration(chessboardSize, frameSize, chessboardSquareMM):
    # Chess board definition
    chessboardSize = chessboardSize
    try:
        assert type(chessboardSize) is tuple
    except AssertionError:
        print('Incorrect chessboardSize, make sure to input a 2 elelements tuple. Example: (7, 4)')
        return 0
    try:
        assert len(chessboardSize) == 2
    except AssertionError:
        print('Incorrect chessboardSize, make sure to input a 2 elelements tuple. Example: (7, 4)')
        return 0

    frameSize = frameSize

    try:
        assert type(frameSize) is tuple
    except AssertionError:
        print('Incorrect frameSize, make sure to input a 2 elelements tuple. Example: (640, 480)')
        return 0
    try:
        assert len(frameSize) == 2
    except AssertionError:
        print('Incorrect frameSize, make sure to input a 2 elelements tuple. Example: (640, 480)')
        return 0

    # Termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare world coordinates
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

    size_of_chessboard_squares_mm = chessboardSquareMM
    objp = objp * size_of_chessboard_squares_mm

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # Get calibration images
    images = glob.glob('./Images/*.jpg')
    if images:
        print("Calibration images acquired. Calculating...")
    else:
        print("No calibration images found.")
        print("Make sure there are images in the ./Images folder or the Images folder exist.")
        return 0

    # Get imgpoints (the corners of the chessboard)
    for image in images:

        img = cv.imread(image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

            # //////////////// COMMENT THIS SECTION IF YOU DON'T WANT TO SEE CALIB IMAGES ///////////
            # cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
            #
            # scale_percent = 60  # percent of original size
            # width = int(img.shape[1] * scale_percent / 100)
            # height = int(img.shape[0] * scale_percent / 100)
            # dim = (width, height)
            #
            # # resize image
            # resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
            # cv.imshow('img', resized)
            # cv.waitKey(100)
            # ///////////////////////////////////////////////////////////////////////////////////////

    cv.destroyAllWindows()

    # Get all the matrices for calculation (cameraMatrix, rvecs and tvecs)
    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
    # print(cameraMatrix)
    # print(rvecs)
    # print(tvecs)

    # Calculate the transformation matrix (homography)
    rotationMatrix = rodrigues_vec_to_rotation_mat(np.squeeze(rvecs[0]))
    extractedData = rotationMatrix[:, [0, 1]]
    deprecatedMatrix = np.c_[np.squeeze(extractedData), np.squeeze(tvecs[0])]
    homographyMatrix = np.squeeze(cameraMatrix) @ deprecatedMatrix
    transformationMatrix = homographyMatrix/(np.squeeze(tvecs[0][2]))
    print('\n World to Pixel Transformation Matrix: \n', transformationMatrix)
    print('\n Pixel to World Transformation Matrix: \n', np.linalg.inv(transformationMatrix))
    path = './calibSaves'
    isExist = os.path.exists(path)
    if isExist:
        # np.save('./calibSaves/WtPMatrix.npy', transformationMatrix)
        np.save('./calibSaves/PtWMatrix.npy', np.linalg.inv(transformationMatrix))
    else:
        os.mkdir(path)
        # np.save('./calibSaves/WtPMatrix.npy', transformationMatrix)
        np.save('./calibSaves/PtWMatrix.npy', np.linalg.inv(transformationMatrix))
    print('\nTransformation matrices saved at ./calibSaves')

    # Test the result on a file (See ./outputs/annotatedFullUpdate.jpg)
    image = cv.imread('./Images/0.jpg')
    annotated_img = copy.deepcopy(image)
    centerPointW = (0, 0, 1)
    centerPointP = transformationMatrix @ centerPointW
    draw_crosshair(annotated_img, (round(centerPointP[0]), round(centerPointP[1])), 40, (0, 0, 255))

    test_XY_2 = (0, chessboardSquareMM, 1)
    for i in range(1, 10):
        t2 = tuple(ti * i for ti in test_XY_2[0:2])
        t2 = (*t2, 1)
        test_xy_2 = transformationMatrix @ t2
        cv.circle(annotated_img, (round(test_xy_2[0]), round(test_xy_2[1])), radius=9, color=(0, 255, 255), thickness=-1)

    test_XY_2 = (chessboardSquareMM, 0, 1)
    for i in range(1, 10):
        t2 = tuple(ti * i for ti in test_XY_2[0:2])
        t2 = (*t2, 1)
        test_xy_2 = transformationMatrix @ t2
        cv.circle(annotated_img, (round(test_xy_2[0]), round(test_xy_2[1])), radius=9, color=(255, 0, 0), thickness=-1)

    path = './outputs'
    isExist = os.path.exists(path)
    if isExist:
        cv.imwrite("./outputs/annotatedFullUpdate.jpg", annotated_img)
    else:
        os.mkdir(path)
        cv.imwrite("./outputs/annotatedFullUpdate.jpg", annotated_img)
    print('\nTest image saved at ./outputs')


def convertPixelToWorldSingle(planeMatrix):

    planeMatrix = (*planeMatrix, 1)  # Append a 1 to tuple
    # print(pixelToWorldMatrix)  # Print T matrix, can uncomment this
    try:
        worldMatrix = pixelToWorldMatrix @ planeMatrix  # Calculate world matrix
    except ValueError:
        print("Wrong input size. Make sure the input form is (x,y)")
    else:
        output = (worldMatrix[0], worldMatrix[1])  # Remove the 1 in matrix
        return output

def convertPixelToWorld(list):
    worldCoordsList = []
    for coords in list:
        planeMatrix = (*coords, 1)  # Append a 1 to tuple
        try:
            worldMatrix = pixelToWorldMatrix @ planeMatrix  # Calculate world matrix
        except ValueError:
            print("Wrong input size. Make sure the input form is (x,y)")
            return 0
        else:
            output = (float(round(Decimal(worldMatrix[0]), 3)), float(round(Decimal(worldMatrix[1]), 3)))  # Remove the 1 in matrix
            worldCoordsList.append(output)
    return worldCoordsList


if __name__ == '__main__':
    testList = [(1302, 1104), (1207, 1268), (1420, 1580), (1284, 1547)]
    print(testList)
    runCalibration((7,4), (640, 480), 22)
    pixelToWorldMatrix = np.load('./calibSaves/PtWMatrix.npy')
    print(convertPixelToWorld(testList))
