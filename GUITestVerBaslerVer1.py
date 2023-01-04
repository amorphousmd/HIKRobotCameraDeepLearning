import math

from Ui_Design import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from Camera_SDK.CamOperation_class import CameraOperation
from MvCameraControl_class import *
from MvErrorDefine_const import *
from CameraParams_header import *
import cv2, imutils, threading
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QTimer
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector
import numpy as np
from centerUtils import detect_center
from centerUtils import detect_center_bbox
import os
import time
import CameraUtils
import TCPIP
import serverUtilities
# pypylon
import pypylon.pylon as py
# plotting for graphs and display of image
import matplotlib.pyplot as plt
# linear algebra and basic math on image matrices
import numpy as np
# OpenCV for image processing functions
import cv2

def TxtWrapBy(start_str, end, all):
    start = all.find(start_str)
    if start >= 0:
        start += len(start_str)
        end = all.find(end, start)
        if end >= 0:
            return all[start:end].strip()


def ToHexStr(num):
    chaDic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
    hexStr = ""
    if num < 0:
        num = num + 2 ** 32
    while num >= 16:
        digit = num % 16
        hexStr = chaDic.get(digit, str(digit)) + hexStr
        num //= 16
    hexStr = chaDic.get(num, str(num)) + hexStr
    return hexStr

def swapTuple2(tuple2):
    try:
        assert type(tuple2) is tuple
    except AssertionError:
        return 0
    try:
        assert len(tuple2) == 2
    except AssertionError:
        return 0
    tempList = list(tuple2)
    swappedList = (tempList[1], tempList[0])
    returnTuple = tuple(swappedList)
    return returnTuple

def rescale(tuple, scale):
    returnTuple = ()
    for elems in tuple:
        returnTuple += (int(elems * scale),)
    return returnTuple

global deviceList
deviceList = MV_CC_DEVICE_INFO_LIST()
cam = MvCamera()
global baslerCam
global nSelCamIndex
nSelCamIndex = 0
global obj_cam_operation
obj_cam_operation = 0
global isOpen
isOpen = False
global isGrabbing
isGrabbing = False
global isCalibMode  # CalibMode check
isCalibMode = True

# Basler camera variables
global baslerCam
baslerCam = 0
global converter
converter = 0

class Logic(QMainWindow, Ui_MainWindow):
    def overwriteLogic(self):
        self.btnRunInferenceVideo.setEnabled(False)
        self.btnRunInference.setEnabled(False)
        self.model_name = 'solov2'  # Default model
        self.run = False
        self.Timer = QTimer()
        self.Timer.timeout.connect(self.trigger_once)
        self.cfg = Config.fromfile('mmdetection/configs/solov2/solov2_light_r18_fpn_3x_coco.py')
        self.cfg.model.mask_head.num_classes = 1
        self.polygonMask = True
        self.ignoreFrame = False

        self.btnEnum.clicked.connect(self.enum_devices)
        self.btnOpen.clicked.connect(self.open_device)
        self.btnClose.clicked.connect(self.close_device)
        self.bnStart.clicked.connect(self.start_grabbing)
        self.bnStop.clicked.connect(self.stop_grabbing)
        # self.bnSave.clicked.connect(self.saveImage)
        self.btnGetParam.clicked.connect(self.get_param)
        self.btnSetParam.clicked.connect(self.set_param)
        self.radioContinueMode.clicked.connect(self.set_continue_mode)
        self.radioTriggerMode.clicked.connect(self.set_software_trigger_mode)
        self.btnLoadCheckpoint.clicked.connect(self.load_model)
        self.btnRunInferenceVideo.clicked.connect(self.run_model)
        self.btnRunCalib.clicked.connect(self.runCameraCalib)
        self.btnLoadCalib.clicked.connect(self.loadCameraCalib)
        self.btnRunInference.clicked.connect(self.runInferenceImage)
        self.btnLoadImage.clicked.connect(self.loadImage)
        self.comboModels.currentIndexChanged.connect(self.select_model)
        self.btnStartCamCalibTest.clicked.connect(self.startCamCalibTest)
        self.btnStartServer.clicked.connect(self.start_server)
        self.btnConnAddr.clicked.connect(self.print_values)
        self.btnSendTCPIP.clicked.connect(self.send_data)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def __init__(self, *args, **kwargs):
        QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)
    def start_server(self):
        serverUtilities.establish_connection()

    def print_values(self):
        print('')
        print(serverUtilities.conn)
        print(serverUtilities.addr)

    def send_data(self):
        if self.btnCustom.isChecked():
            str_data = self.textEdit.toPlainText()
            list_data = str_data.split(" ")
            temp_tuple = ()
            send_data = []
            i = 0
            for str_num in list_data:
                int_num = int(str_num)
                if i == 1:
                    i = 0
                    temp_tuple = temp_tuple + (int_num, )
                    send_data.append(temp_tuple)
                    temp_tuple = ()
                else:
                    temp_tuple = temp_tuple + (int_num, )
                    i += 1

        if self.btnOption2.isChecked():
            send_data = []
        if self.btnOption3.isChecked():
            send_data = [(1, 2), (3, 4)]
        if self.btnOption4.isChecked():
            send_data = [(1, 2), (3, 4), (5, 6)]

        thread = threading.Thread(target=serverUtilities.send_data_thread, args=[send_data])
        thread.start()

    def startCamCalibTest(self):
        self.filename = QFileDialog.getOpenFileName(directory="./calibImages20221020")[0]
        self.image = cv2.imread(self.filename)
        self.set_image(self.image)
        self.displayLabel.mousePressEvent = self.getPos
        self.calculateDistance = False
        self.previousX = 0
        self.previousY = 0

    def getPos(self, event):
        if not self.calculateDistance:
            # Swapping x and y according to the convention
            y = round(event.pos().x() /1000 * 2592)
            x = round(event.pos().y() /1000 * 2592)
            self.editPixCoordX.setText(str(x))
            self.editPixCoordY.setText(str(y))
            convertTuple = (x, y)
            convertedCoords = CameraUtils.convertPixelToWorldSingle(convertTuple)
            # print(convertedCoords[0], convertedCoords[1])
            self.previousX = convertedCoords[0]
            self.previousY = convertedCoords[1]
            self.editWorldCoordX.setText(str(convertedCoords[0]))
            self.editPixelCoordY.setText(str(convertedCoords[1]))
            self.calculateDistance = True
        else:
            y = round(event.pos().x() / 1000 * 2592)
            x = round(event.pos().y() / 1000 * 2592)
            self.editPixCoordX.setText(str(x))
            self.editPixCoordY.setText(str(y))
            convertTuple = (x, y)
            convertedCoords = CameraUtils.convertPixelToWorldSingle(convertTuple)
            # print(convertedCoords[0], convertedCoords[1])
            self.editWorldCoordX.setText(str(convertedCoords[0]))
            self.editPixelCoordY.setText(str(convertedCoords[1]))
            distance = math.sqrt((convertedCoords[0] - self.previousX) ** 2 + (convertedCoords[1] - self.previousY) ** 2)
            self.editWorldLength.setText(str(distance))
            # print(distance)
            self.calculateDistance = False

    def loadCameraCalib(self):
        filename = CameraUtils.loadCalibration()
        if not filename:
            return 0
        self.editLoadedCalib.setText(filename)

    def runCameraCalib(self):
        CameraUtils.runCalibration((10, 7), (2592, 1944), 25)

    def saveImage(self):
        self.filename = QFileDialog.getSaveFileName(filter="JPG(*.jpg);;PNG(*.png);;TIFF(*.tiff);;BMP(*.bmp)")[0]
        self.imgSave = cv2.cvtColor(self.imgSave, cv2.COLOR_BGR2RGB)
        cv2.imwrite(self.filename, self.imgSave)
        print('Image saved as:', self.filename)

    def select_model(self, i):
        if i == 0:
            print('Model: SOLOv2')
            self.model_name = 'solov2'
            self.cfg = Config.fromfile('mmdetection/configs/solov2/solov2_light_r18_fpn_3x_coco.py')
            self.cfg.model.mask_head.num_classes = 1
            self.polygonMask = True
        if i == 1:
            pass
        if i == 2:
            print('Model: YOLOX-s')
            self.model_name = 'solov2'
            self.cfg = Config.fromfile('mmdetection/configs/yolox/yolox_s_8x8_300e_coco.py')
            self.cfg.model.bbox_head.num_classes = 1
            self.polygonMask = False

    def load_model(self):
        self.checkpoint = QFileDialog.getOpenFileName()[0]
        print('Weights loaded: ' + self.checkpoint + '\n')
        model = build_detector(self.cfg.model)
        checkpoint = load_checkpoint(model, self.checkpoint, map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.cfg = self.cfg
        model.to('cuda')
        model.eval()
        self.model = model
        self.btnRunInferenceVideo.setEnabled(True)
        self.btnRunInference.setEnabled(True)

    def run_model(self):
        self.scaleFactor = float(self.editScoreThreshold_3.toPlainText())
        self.run = True

    def detect(self, image, score_thr_value, center=True):

        self.time_start = time.time()

        result = inference_detector(self.model, image)
        displayLabel = self.model.show_result(
            image,
            result,
            score_thr=score_thr_value,
            show=False,
            wait_time=0,
            win_name='result',
            bbox_color=None,
            text_color=(200, 200, 200),
            mask_color=None,
            out_file=None)
        self.time_detect = time.time() - self.time_start
        self.lblInferenceTime.setText(str(self.time_detect))
        if self.polygonMask:
            center_list = detect_center(image, result, score_thr_value)
            print(center_list)
            print('\nPixel Coordinates:\n')
            print(center_list)
            print('\nWorld Coordinates:\n')
            print(CameraUtils.convertPixelToWorld(center_list))
            # TCPIP.sendData(CameraUtils.convertPixelToWorld(center_list))
            thread = threading.Thread(target=serverUtilities.send_data_thread, args=[CameraUtils.convertPixelToWorld(center_list)])
            thread.start()

            for center in center_list:
                displayLabel = cv2.circle(displayLabel, center, 4, (0, 0, 255), -1)
        else:
            center_list = detect_center_bbox(result, score_thr_value, self.scaleFactor)
            print(center_list)
            print('\nPixel Coordinates:\n')
            print(center_list)
            print('\nWorld Coordinates:\n')
            print(CameraUtils.convertPixelToWorld(center_list))
            # TCPIP.sendData(CameraUtils.convertPixelToWorld(center_list))
            thread = threading.Thread(target=serverUtilities.send_data_thread, args=[CameraUtils.convertPixelToWorld(center_list)])
            thread.start()

            for center in center_list:
                rescaledCenter = rescale(swapTuple2(center), self.scaleFactor)
                displayLabel = cv2.circle(displayLabel, rescaledCenter, 4, (0, 0, 255), -1)
        # self.set_image(displayLabel)
        return displayLabel

    def set_img_show(self, image):
        """
        Display function that doesn't invert color
        """
        self.tmp = image
        if image is not None:
            image = imutils.resize(image, width=1000)
            frame = image
            # frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            self.displayLabel.setPixmap(QtGui.QPixmap.fromImage(image))

    def xFunc(event):
        global nSelCamIndex
        nSelCamIndex = TxtWrapBy("[", "]", ui.comboDevices.itemData(ui.comboDevices.currentIndex()))

        # en:enum devices

    def enum_devices(self):
        global baslerCam
        global converter
        tlf = py.TlFactory.GetInstance()
        # print(tlf)
        devices = tlf.EnumerateDevices()
        # print(devices)
        devList = []
        i = 0
        for d in devices:
            print(d.GetModelName(), d.GetSerialNumber())
            devList.append("[" + str(i) + "] " + d.GetModelName() + " "+ d.GetSerialNumber())
        baslerCam = py.InstantCamera(tlf.CreateDevice(devices[0]))
        baslerCam = py.InstantCamera(tlf.CreateFirstDevice())


        converter = py.ImageFormatConverter()
        converter.OutputPixelFormat = py.PixelType_BGR8packed
        converter.OutputBitAlignment = py.OutputBitAlignment_MsbAligned

        self.comboDevices.clear()
        self.comboDevices.addItems(devList)
        self.comboDevices.setCurrentIndex(0)
        # en:open device

    def open_device(self):
        global isOpen
        baslerCam.Open()
        isOpen = True
        print("Camera Opened")
        self.radioContinueMode.setChecked(1)


        # en:Start grab image

    def start_grabbing(self):
        global obj_cam_operation
        global isGrabbing
        global cam
        global baslerCam
        # ret = obj_cam_operation.Start_grabbing(ui.widgetDisplay.winId())
        # ret = cam.MV_CC_StartGrabbing()
        # if ret != 0:
        #     strError = "Start grabbing failed ret:" + ToHexStr(ret)
        #     QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        # else:
        isGrabbing = True
        baslerCam.StartGrabbing(py.GrabStrategy_LatestImageOnly)
        if self.radioTriggerMode.isChecked():
            print(self.editTimeTrigger.toPlainText())
            isGrabbing = True
            self.Timer.start(int(self.editTimeTrigger.toPlainText()))
            self.trigger_once()
            return
        grab_thread = threading.Thread(target=self.thread)
        grab_thread.start()


    def thread(self):
        global obj_cam_operation
        while True:
            grabResult = baslerCam.RetrieveResult(5000, py.TimeoutHandling_ThrowException)

            try:
                check = grabResult.GrabSucceeded()
            except:
                break
            if check:
                # Access the image data
                image = converter.Convert(grabResult)
                self.img = image.GetArray()
                self.set_image(self.img)

            grabResult.Release()
            if isGrabbing == False:
                break


        # en:Stop grab image

    def stop_grabbing(self):
        global baslerCam
        global isGrabbing
        baslerCam.StopGrabbing()
        isGrabbing = False
        # if ret != 0:
        #     strError = "Stop grabbing failed ret:" + ToHexStr(ret)
        #     QMessageBox.warning(QMainWindow(), "Error", strError, QMessageBox.Ok)
        # else:
        #     self.Timer.stop()
        #     isGrabbing = False
        #     self.enable_controls()

        # Close device

    def close_device(self):
        global isOpen
        global isGrabbing
        global obj_cam_operation
        global baslerCam

        if isOpen:
            baslerCam.Close()
            isOpen = False

        isGrabbing = False

    def set_continue_mode(self):
        self.Timer.stop()
        # global is_trigger_mode
        # strError = None
        #
        # ret = obj_cam_operation.Set_trigger_mode(False)
        # if ret != 0:
        #     strError = "Set continue mode failed ret:" + ToHexStr(ret) + " mode is " + str(is_trigger_mode)
        #     QMessageBox.warning(QMainWindow(), "Error", strError, QMessageBox.Ok)
        # else:
        #     self.radioContinueMode.setChecked(True)
        #     self.radioTriggerMode.setChecked(False)
        #     # ui.bnSoftwareTrigger.setEnabled(False)
        pass
        # en:set software trigger mode

    def set_software_trigger_mode(self):
        global isOpen
        global isGrabbing
        global obj_cam_operation

        # ret = obj_cam_operation.Set_trigger_mode(True)
        # if ret != 0:
        #     strError = "Set trigger mode failed ret:" + ToHexStr(ret)
        #     QMessageBox.warning(QMainWindow(), "Error", strError, QMessageBox.Ok)
        # else:
        #
        #     self.radioContinueMode.setChecked(False)
        #     self.radioTriggerMode.setChecked(True)
        #     # ui.bnSoftwareTrigger.setEnabled(isGrabbing)

        # en:set trigger software

    def trigger_once(self):
        grabResult = baslerCam.RetrieveResult(5000, py.TimeoutHandling_ThrowException)

        try:
            check = grabResult.GrabSucceeded()
        except:
            return
        if check:
            # Access the image data
            image = converter.Convert(grabResult)
            self.img = image.GetArray()
            self.set_image(self.img)

        grabResult.Release()
        # ret = obj_cam_operation.Trigger_once()
        # if ret != 0:
        #     # strError = "TriggerSoftware failed ret:" + ToHexStr(ret)
        #     # QMessageBox.warning(QMainWindow(), "Error", strError, QMessageBox.Ok)
        #     print('TriggerSoffware failed ret:' + ToHexStr(ret))

        # en:save image

    def save_bmp(self):
        ret = obj_cam_operation.Save_Bmp()
        if ret != MV_OK:
            strError = "Save BMP failed ret:" + ToHexStr(ret)
            QMessageBox.warning(QMainWindow(), "Error", strError, QMessageBox.Ok)
        else:
            print("Save image success")

        # en:get param

    def get_param(self):
        ret = obj_cam_operation.Get_parameter()
        if ret != MV_OK:
            strError = "Get param failed ret:" + ToHexStr(ret)
            QMessageBox.warning(QMainWindow(), "Error", strError, QMessageBox.Ok)
        else:
            self.edtExposureTime.setText("{0:.2f}".format(obj_cam_operation.exposure_time))
            self.edtGain.setText("{0:.2f}".format(obj_cam_operation.gain))
            self.edtFrameRate.setText("{0:.2f}".format(obj_cam_operation.frame_rate))

        # en:set param

    def set_param(self):
        frame_rate = self.edtFrameRate.toPlainText()
        exposure = self.edtExposureTime.toPlainText()
        gain = self.edtGain.toPlainText()
        ret = obj_cam_operation.Set_parameter(frame_rate, exposure, gain)
        if ret != MV_OK:
            strError = "Set param failed ret:" + ToHexStr(ret)
            QMessageBox.warning(QMainWindow(), "Error", strError, QMessageBox.Ok)

        return MV_OK

    def get_image(self):
        global obj_cam_operation
        img = obj_cam_operation.get_np_image()
        return img

        # en:set enable status

    def enable_controls(self):
        global isGrabbing
        global isOpen

        # ui.groupGrab.setEnabled(isOpen)
        # ui.groupParam.setEnabled(isOpen)

        self.btnOpen.setEnabled(not isOpen)
        self.btnClose.setEnabled(isOpen)

        self.bnStart.setEnabled(isOpen and (not isGrabbing))
        self.bnStop.setEnabled(isOpen and isGrabbing)

    def loadImage(self):
        self.filename = QFileDialog.getOpenFileName(directory="C:/Users/LAPTOP/Desktop/Pics")[0]
        self.image = cv2.imread(self.filename)
        self.set_image(self.image)

    def set_image(self, image):
        """ This function will take image input and resize it
            only for display purpose and convert it to QImage
            to set at the label.
        """
        self.tmp = image
        image = imutils.resize(image, width=1000)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.displayLabel.setPixmap(QtGui.QPixmap.fromImage(image))

    def runInferenceImage(self):
        self.time_start = time.time()
        if self.editScoreThreshold.toPlainText() == "":
            score_threshold = 0.5  # Default threshold Value = 0.5
        else:
            score_threshold = float(self.editScoreThreshold.toPlainText())
        temp = self.image
        width = int(temp.shape[1] * 50 / 100)
        height = int(temp.shape[0] * 50 / 100)
        dim = (width, height)

        # resize image
        temp = cv2.resize(temp, dim, interpolation=cv2.INTER_AREA)
        result = inference_detector(self.model, temp)
        displayLabel = self.model.show_result(
            temp,
            result,
            score_thr=score_threshold,
            show=False,
            wait_time=0,
            win_name='result',
            bbox_color=None,
            text_color=(200, 200, 200),
            mask_color=None,
            out_file=None)
        self.time_detect = time.time() - self.time_start
        self.editInferenceTimeImage.setText(str(self.time_detect))

        if self.polygonMask:
            center_list = detect_center(self.image, result, score_threshold)
            print(center_list)
            print('\nPixel Coordinates:\n')
            print(center_list)
            print('\nWorld Coordinates:\n')
            print(CameraUtils.convertPixelToWorld(center_list))
            # TCPIP.sendData(CameraUtils.convertPixelToWorld(center_list))
            thread = threading.Thread(target=serverUtilities.send_data_thread, args=[center_list])
            thread.start()

            for center in center_list:
                displayLabel = cv2.circle(displayLabel, swapTuple2(center), 10, (0, 0, 255), -1)
        else:
            center_list = detect_center_bbox(result, 0.8, 1)
            print(center_list)
            print('\nPixel Coordinates:\n')
            print(center_list)
            print('\nWorld Coordinates:\n')
            print(CameraUtils.convertPixelToWorld(center_list))
            # TCPIP.sendData(CameraUtils.convertPixelToWorld(center_list))
            thread = threading.Thread(target=serverUtilities.send_data_thread, args=[center_list])
            thread.start()

            for center in center_list:
                rescaledCenter = rescale(swapTuple2(center), 1)
                displayLabel = cv2.circle(displayLabel, rescaledCenter, 4, (0, 0, 255), -1)
        self.set_image(displayLabel)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Logic()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())