import cv2 as cv
import copy
import numpy as np
import os
import math
# Test the result on a file (See ./outputs/annotatedFullUpdate.jpg)

def draw_crosshair(image, center, width, color):
    cv.line(image, (center[0] - width // 2, center[1]), (center[0] + width // 2, center[1]), color, 9)
    cv.line(image, (center[0], center[1] - width // 2), (center[0], center[1] + width // 2), color, 9)

transformationMatrixpre = np.load('calibSaves/PtWMatrix.npy')
transformationMatrix = np.linalg.inv(transformationMatrixpre)
# print(transformationMatrix)

centerPointW = (0, 0, 1)
centerDisplacementP = transformationMatrix @ centerPointW
print(centerDisplacementP)
testPoint = (500, 0, 1)
test_xy_angle = transformationMatrix @ testPoint
print(test_xy_angle)
rotationTriangle = test_xy_angle - centerDisplacementP
print(rotationTriangle)
theta1 = math.atan2(rotationTriangle[1], rotationTriangle[0])
print('answer:', theta1)

addedVector = np.array([centerDisplacementP[0], centerDisplacementP[1], 0])
addedVector = addedVector.reshape((-1, 1))
# print(addedVector)
leftMatrix = np.array([[0, 0], [0, 0], [0,0]])
# print(leftMatrix)
addedMatrix = np.c_[leftMatrix, addedVector]              # add a column
# print(addedMatrix)
transformationMatrix = transformationMatrix - addedMatrix

theta = 10
theta = -theta1
rotationSquare = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
print(rotationSquare)
rightcolumn = np.array([[0], [0]])
rotationRectangle = np.c_[rotationSquare, rightcolumn]
print(rotationRectangle)
bottomRow = [0, 0, 1]
rotationMatrix = np.vstack([rotationRectangle, bottomRow])
print(rotationMatrix)
transformationMatrix = transformationMatrix @ rotationMatrix

np.save('./calibSaves/PtWMatrixModified.npy', np.linalg.inv(transformationMatrix))


image = cv.imread('./Images/0.jpg')
annotated_img = copy.deepcopy(image)
centerPointW = (0, 0, 1)
centerPointP = transformationMatrix @ centerPointW
draw_crosshair(annotated_img, (round(centerPointP[0]), round(centerPointP[1])), 40, (0, 0, 255))

cv.circle(annotated_img, (round(test_xy_angle[0]), round(test_xy_angle[1])), radius=20, color=(255, 255, 255), thickness=-1)

test_XY_2 = (0, 25, 1)
for i in range(1, 20):
    t2 = tuple(ti * i for ti in test_XY_2[0:2])
    t2 = (*t2, 1)
    test_xy_2 = transformationMatrix @ t2
    cv.circle(annotated_img, (round(test_xy_2[0]), round(test_xy_2[1])), radius=9, color=(0, 255, 255), thickness=-1)

test_XY_2 = (25, 0, 1)
for i in range(1, 20):
    t2 = tuple(ti * i for ti in test_XY_2[0:2])
    t2 = (*t2, 1)
    test_xy_2 = transformationMatrix @ t2
    cv.circle(annotated_img, (round(test_xy_2[0]), round(test_xy_2[1])), radius=9, color=(255, 0, 0), thickness=-1)

path = './outputs'
isExist = os.path.exists(path)
if isExist:
    cv.imwrite("./outputs/TMatrixModifyTest.jpg", annotated_img)
else:
    os.mkdir(path)
    cv.imwrite("./outputs/TMatrixModifyTest.jpg", annotated_img)
print('\nTest image saved at ./outputs')