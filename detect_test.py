# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'detect_test.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

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
from mmdet.apis.inference import inference_detector
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector
import numpy as np
from model import detect_center
import time

def TxtWrapBy(start_str, end, all):
    start = all.find(start_str)
    if start >= 0:
        start += len(start_str)
        end = all.find(end, start)
        if end >= 0:
            return all[start:end].strip()


# 将返回的错误码转换为十六进制显示
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



global deviceList
deviceList = MV_CC_DEVICE_INFO_LIST()
global cam
cam = MvCamera()
global nSelCamIndex
nSelCamIndex = 0
global obj_cam_operation
obj_cam_operation = 0
global isOpen
isOpen = False
global isGrabbing
isGrabbing = False
global isCalibMode  # 是否是标定模式（获取原始图像）
isCalibMode = True



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1339, 612)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 10, 771, 551))
        self.groupBox.setObjectName("groupBox")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(260, 200, 63, 20))
        self.label.setText("")
        self.label.setObjectName("label")
        self.Im_show = QtWidgets.QLabel(self.groupBox)
        self.Im_show.setGeometry(QtCore.QRect(10, 20, 741, 511))
        self.Im_show.setText("")
        self.Im_show.setObjectName("Im_show")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(800, 20, 261, 491))
        self.groupBox_2.setObjectName("groupBox_2")
        self.ComboDevices = QtWidgets.QComboBox(self.groupBox_2)
        self.ComboDevices.setGeometry(QtCore.QRect(20, 30, 231, 31))
        self.ComboDevices.setObjectName("ComboDevices")
        self.bnEnum = QtWidgets.QPushButton(self.groupBox_2)
        self.bnEnum.setGeometry(QtCore.QRect(20, 70, 231, 31))
        self.bnEnum.setObjectName("bnEnum")
        self.bnOpen = QtWidgets.QPushButton(self.groupBox_2)
        self.bnOpen.setGeometry(QtCore.QRect(20, 110, 91, 31))
        self.bnOpen.setObjectName("bnOpen")
        self.bnClose = QtWidgets.QPushButton(self.groupBox_2)
        self.bnClose.setGeometry(QtCore.QRect(160, 110, 91, 31))
        self.bnClose.setObjectName("bnClose")
        self.radioContinueMode = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioContinueMode.setGeometry(QtCore.QRect(20, 150, 110, 24))
        self.radioContinueMode.setObjectName("radioContinueMode")
        self.radioTriggerMode = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioTriggerMode.setGeometry(QtCore.QRect(140, 150, 110, 24))
        self.radioTriggerMode.setObjectName("radioTriggerMode")
        self.bnStart = QtWidgets.QPushButton(self.groupBox_2)
        self.bnStart.setGeometry(QtCore.QRect(20, 220, 93, 29))
        self.bnStart.setObjectName("bnStart")
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_2.setGeometry(QtCore.QRect(20, 180, 101, 21))
        self.label_2.setObjectName("label_2")
        self.Text_Time_trigger = QtWidgets.QLineEdit(self.groupBox_2)
        self.Text_Time_trigger.setGeometry(QtCore.QRect(130, 180, 111, 31))
        self.Text_Time_trigger.setObjectName("Text_Time_trigger")
        self.bnStop = QtWidgets.QPushButton(self.groupBox_2)
        self.bnStop.setGeometry(QtCore.QRect(140, 220, 93, 29))
        self.bnStop.setObjectName("bnStop")
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox_2)
        self.groupBox_3.setGeometry(QtCore.QRect(20, 270, 231, 201))
        self.groupBox_3.setObjectName("groupBox_3")
        self.label_3 = QtWidgets.QLabel(self.groupBox_3)
        self.label_3.setGeometry(QtCore.QRect(10, 30, 101, 21))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.groupBox_3)
        self.label_4.setGeometry(QtCore.QRect(10, 70, 101, 21))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.groupBox_3)
        self.label_5.setGeometry(QtCore.QRect(10, 110, 101, 21))
        self.label_5.setObjectName("label_5")
        self.bnGetParam = QtWidgets.QPushButton(self.groupBox_3)
        self.bnGetParam.setGeometry(QtCore.QRect(10, 150, 93, 29))
        self.bnGetParam.setObjectName("bnGetParam")
        self.bnSetParam = QtWidgets.QPushButton(self.groupBox_3)
        self.bnSetParam.setGeometry(QtCore.QRect(120, 150, 93, 29))
        self.bnSetParam.setObjectName("bnSetParam")
        self.edtExposureTime = QtWidgets.QLineEdit(self.groupBox_3)
        self.edtExposureTime.setGeometry(QtCore.QRect(120, 30, 91, 31))
        self.edtExposureTime.setObjectName("edtExposureTime")
        self.edtGain = QtWidgets.QLineEdit(self.groupBox_3)
        self.edtGain.setGeometry(QtCore.QRect(120, 70, 91, 31))
        self.edtGain.setObjectName("edtGain")
        self.edtFrameRate = QtWidgets.QLineEdit(self.groupBox_3)
        self.edtFrameRate.setGeometry(QtCore.QRect(120, 110, 91, 31))
        self.edtFrameRate.setObjectName("edtFrameRate")
        self.Combo_model = QtWidgets.QGroupBox(self.centralwidget)
        self.Combo_model.setGeometry(QtCore.QRect(1070, 30, 241, 261))
        self.Combo_model.setObjectName("Combo_model")
        self.label_13 = QtWidgets.QLabel(self.Combo_model)
        self.label_13.setGeometry(QtCore.QRect(10, 40, 63, 20))
        self.label_13.setObjectName("label_13")
        self.ComboDevices_2 = QtWidgets.QComboBox(self.Combo_model)
        self.ComboDevices_2.setGeometry(QtCore.QRect(70, 30, 141, 31))
        self.ComboDevices_2.setObjectName("ComboDevices_2")
        self.ComboDevices_2.addItem("")
        self.bnLoad_model = QtWidgets.QPushButton(self.Combo_model)
        self.bnLoad_model.setGeometry(QtCore.QRect(10, 70, 211, 31))
        self.bnLoad_model.setObjectName("bnLoad_model")
        self.label_18 = QtWidgets.QLabel(self.Combo_model)
        self.label_18.setGeometry(QtCore.QRect(10, 120, 63, 20))
        self.label_18.setObjectName("label_18")
        self.Text_time_detect = QtWidgets.QLineEdit(self.Combo_model)
        self.Text_time_detect.setGeometry(QtCore.QRect(90, 120, 131, 31))
        self.Text_time_detect.setObjectName("Text_time_detect")
        self.label_19 = QtWidgets.QLabel(self.Combo_model)
        self.label_19.setGeometry(QtCore.QRect(10, 170, 71, 21))
        self.label_19.setObjectName("label_19")
        self.Text_score_thr = QtWidgets.QLineEdit(self.Combo_model)
        self.Text_score_thr.setGeometry(QtCore.QRect(90, 170, 131, 31))
        self.Text_score_thr.setObjectName("Text_score_thr")
        self.bnRun = QtWidgets.QPushButton(self.Combo_model)
        self.bnRun.setGeometry(QtCore.QRect(10, 210, 93, 29))
        self.bnRun.setObjectName("bnRun")
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(1070, 300, 241, 71))
        self.groupBox_4.setObjectName("groupBox_4")
        self.label_20 = QtWidgets.QLabel(self.groupBox_4)
        self.label_20.setGeometry(QtCore.QRect(10, 30, 63, 20))
        self.label_20.setObjectName("label_20")
        self.bnLoadCalib = QtWidgets.QPushButton(self.groupBox_4)
        self.bnLoadCalib.setGeometry(QtCore.QRect(90, 30, 111, 31))
        self.bnLoadCalib.setObjectName("bnLoadCalib")
        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_5.setGeometry(QtCore.QRect(1070, 380, 241, 131))
        self.groupBox_5.setObjectName("groupBox_5")
        self.label_21 = QtWidgets.QLabel(self.groupBox_5)
        self.label_21.setGeometry(QtCore.QRect(10, 30, 41, 21))
        self.label_21.setObjectName("label_21")
        self.text_Send = QtWidgets.QLineEdit(self.groupBox_5)
        self.text_Send.setGeometry(QtCore.QRect(60, 30, 171, 91))
        self.text_Send.setObjectName("text_Send")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1339, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.model_name = 'solov2'
        self.run = False
        self.Timer = QTimer()

        self.Timer.timeout.connect(self.trigger_once)
        self.bnEnum.clicked.connect(self.enum_devices)
        self.bnOpen.clicked.connect(self.open_device)
        self.bnClose.clicked.connect(self.close_device)
        self.bnStart.clicked.connect(self.start_grabbing)
        self.bnStop.clicked.connect(self.stop_grabbing)
        self.bnRun.clicked.connect(self.run_model)

        self.radioTriggerMode.clicked.connect(self.set_software_trigger_mode)
        self.radioContinueMode.clicked.connect(self.set_continue_mode)

        self.bnGetParam.clicked.connect(self.get_param)
        self.bnSetParam.clicked.connect(self.set_param)
        self.bnLoad_model.clicked.connect(self.load_model)
        self.ComboDevices_2.currentIndexChanged.connect(self.select_model)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def select_model(self, i):
        if i == 0:
            self.model_name = 'solov2'

    def load_model(self):
        checkpoint = QFileDialog.getOpenFileName()[0]
        cfg = Config.fromfile('mmdetection/configs/solov2/solov2_light_r18_fpn_3x_coco.py')
        cfg.model.mask_head.num_classes = 1

        model = build_detector(cfg.model)
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.cfg = cfg
        model.to('cuda')
        model.eval()
        self.model = model


    def run_model(self):
        self.run = True



    def detect(self, image, score_thr_value, center=True):

        self.time_start = time.time()

        result = inference_detector(self.model, image)
        img_show = self.model.show_result(
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
        if center:
            center_list = detect_center(img_show, result, score_thr_value)
            for center_point in center_list:
                img_show = cv2.circle(img_show, center_point, 10, (255, 0, 0), -1)

        self.time_detect = time.time() - self.time_start
        self.Text_time_detect.setText(str(self.time_detect))

        return img_show


    def set_img_show(self,image):
        """ This function will take image input and resize it
            only for display purpose and convert it to QImage
            to set at the label.
        """
        self.tmp = image
        if image is not None:
            image = imutils.resize(image,width=640)
            frame = image
            # frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = QImage(frame, frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)
            self.Im_show.setPixmap(QtGui.QPixmap.fromImage(image))


    def xFunc(event):
        global nSelCamIndex
        nSelCamIndex = TxtWrapBy("[", "]", ui.ComboDevices.get())

        # ch:枚举相机 | en:enum devices


    def enum_devices(self):
        global deviceList
        global obj_cam_operation

        deviceList = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)
        if ret != 0:
            strError = "Enum devices fail! ret = :" + ToHexStr(ret)
            QMessageBox.warning(QMainWindow(), "Error", strError, QMessageBox.Ok)
            return ret

        if deviceList.nDeviceNum == 0:
            QMessageBox.warning(QMainWindow(), "Info", "Find no device", QMessageBox.Ok)
            return ret
        print("Find %d devices!" % deviceList.nDeviceNum)

        devList = []
        for i in range(0, deviceList.nDeviceNum):
            mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                print("\ngige device: [%d]" % i)
                chUserDefinedName = ""
                for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chUserDefinedName:
                    if 0 == per:
                        break
                    chUserDefinedName = chUserDefinedName + chr(per)
                print("device user define name: %s" % chUserDefinedName)

                chModelName = ""
                for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                    if 0 == per:
                        break
                    chModelName = chModelName + chr(per)

                print("device model name: %s" % chModelName)

                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
                devList.append(
                    "[" + str(i) + "]GigE: " + chUserDefinedName + " " + chModelName + "(" + str(nip1) + "." + str(
                        nip2) + "." + str(nip3) + "." + str(nip4) + ")")
            elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                print("\nu3v device: [%d]" % i)
                chUserDefinedName = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chUserDefinedName:
                    if per == 0:
                        break
                    chUserDefinedName = chUserDefinedName + chr(per)
                print("device user define name: %s" % chUserDefinedName)

                chModelName = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                    if 0 == per:
                        break
                    chModelName = chModelName + chr(per)
                print("device model name: %s" % chModelName)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber = strSerialNumber + chr(per)
                print("user serial number: %s" % strSerialNumber)
                devList.append("[" + str(i) + "]USB: " + chUserDefinedName + " " + chModelName
                               + "(" + str(strSerialNumber) + ")")

        self.ComboDevices.clear()
        self.ComboDevices.addItems(devList)
        self.ComboDevices.setCurrentIndex(0)

        # ch:打开相机 | en:open device


    def open_device(self):
        global deviceList
        global nSelCamIndex
        global obj_cam_operation
        global isOpen
        if isOpen:
            QMessageBox.warning(QMainWindow(), "Error", 'Camera is Running!', QMessageBox.Ok)
            return MV_E_CALLORDER

        nSelCamIndex = ui.ComboDevices.currentIndex()
        if nSelCamIndex < 0:
            QMessageBox.warning(QMainWindow(), "Error", 'Please select a camera!', QMessageBox.Ok)
            return MV_E_CALLORDER

        obj_cam_operation = CameraOperation(cam, deviceList, nSelCamIndex)
        ret = obj_cam_operation.Open_device()
        if 0 != ret:
            strError = "Open device failed ret:" + ToHexStr(ret)
            QMessageBox.warning(QMainWindow(), "Error", strError, QMessageBox.Ok)
            isOpen = False
        else:
            self.set_software_trigger_mode()

            self.get_param()

            isOpen = True
            self.enable_controls()

        # ch:开始取流 | en:Start grab image


    def start_grabbing(self):
        global obj_cam_operation
        global isGrabbing
        global cam
        # ret = obj_cam_operation.Start_grabbing(ui.widgetDisplay.winId())
        # ret = cam.MV_CC_StartGrabbing()
        # if ret != 0:
        #     strError = "Start grabbing failed ret:" + ToHexStr(ret)
        #     QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        # else:
        if self.radioTriggerMode.isChecked():
            print(self.Text_Time_trigger.text())
            isGrabbing = True
            self.enable_controls()
            self.Timer.start(int(self.Text_Time_trigger.text()))
        theard = threading.Thread(target=self.thread)
        theard.start()


    def thread(self):
        global obj_cam_operation
        while True:
            img = obj_cam_operation.get_np_image()

            if self.run:
                score_thr = float(self.Text_score_thr.text())
                img = self.detect(img, score_thr)
            self.set_img_show(img)
            if isGrabbing == False:
                break

        # ch:停止取流 | en:Stop grab image


    def stop_grabbing(self):
        global obj_cam_operation
        global isGrabbing
        ret = obj_cam_operation.Stop_grabbing()
        self.thread()
        isGrabbing = False
        if ret != 0:
            strError = "Stop grabbing failed ret:" + ToHexStr(ret)
            QMessageBox.warning(QMainWindow(), "Error", strError, QMessageBox.Ok)
        else:
            self.Timer.stop()
            isGrabbing = False
            self.enable_controls()

        # ch:关闭设备 | Close device



    def close_device(self):
        global isOpen
        global isGrabbing
        global obj_cam_operation

        if isOpen:
            obj_cam_operation.Close_device()
            isOpen = False

        isGrabbing = False

        self.enable_controls()

        # ch:设置触发模式 | en:set trigger mode


    def set_continue_mode(self):
        global is_trigger_mode
        strError = None

        ret = obj_cam_operation.Set_trigger_mode(False)
        if ret != 0:
            strError = "Set continue mode failed ret:" + ToHexStr(ret) + " mode is " + str(is_trigger_mode)
            QMessageBox.warning(QMainWindow(), "Error", strError, QMessageBox.Ok)
        else:
            self.radioContinueMode.setChecked(True)
            self.radioTriggerMode.setChecked(False)
            # ui.bnSoftwareTrigger.setEnabled(False)

        # ch:设置软触发模式 | en:set software trigger mode


    def set_software_trigger_mode(self):
        global isOpen
        global isGrabbing
        global obj_cam_operation


        ret = obj_cam_operation.Set_trigger_mode(True)
        if ret != 0:
            strError = "Set trigger mode failed ret:" + ToHexStr(ret)
            QMessageBox.warning(QMainWindow(), "Error", strError, QMessageBox.Ok)
        else:

            self.radioContinueMode.setChecked(False)
            self.radioTriggerMode.setChecked(True)
            # ui.bnSoftwareTrigger.setEnabled(isGrabbing)

        # ch:设置触发命令 | en:set trigger software


    def trigger_once(self):
        ret = obj_cam_operation.Trigger_once()
        if ret != 0:
            strError = "TriggerSoftware failed ret:" + ToHexStr(ret)
            QMessageBox.warning(QMainWindow(), "Error", strError, QMessageBox.Ok)

        # ch:存图 | en:save image


    def save_bmp(self):
        ret = obj_cam_operation.Save_Bmp()
        if ret != MV_OK:
            strError = "Save BMP failed ret:" + ToHexStr(ret)
            QMessageBox.warning(QMainWindow(), "Error", strError, QMessageBox.Ok)
        else:
            print("Save image success")

        # ch: 获取参数 | en:get param


    def get_param(self):
        ret = obj_cam_operation.Get_parameter()
        if ret != MV_OK:
            strError = "Get param failed ret:" + ToHexStr(ret)
            QMessageBox.warning(QMainWindow(), "Error", strError, QMessageBox.Ok)
        else:
            self.edtExposureTime.setText("{0:.2f}".format(obj_cam_operation.exposure_time))
            self.edtGain.setText("{0:.2f}".format(obj_cam_operation.gain))
            self.edtFrameRate.setText("{0:.2f}".format(obj_cam_operation.frame_rate))

        # ch: 设置参数 | en:set param


    def set_param(self):
        frame_rate = self.edtFrameRate.text()
        exposure = self.edtExposureTime.text()
        gain = self.edtGain.text()
        ret = obj_cam_operation.Set_parameter(frame_rate, exposure, gain)
        if ret != MV_OK:
            strError = "Set param failed ret:" + ToHexStr(ret)
            QMessageBox.warning(QMainWindow(), "Error", strError, QMessageBox.Ok)

        return MV_OK


    def get_image(self):
        global obj_cam_operation
        img = obj_cam_operation.get_np_image()
        return img

        # ch: 设置控件状态 | en:set enable status


    def enable_controls(self):
        global isGrabbing
        global isOpen

        # 先设置group的状态，再单独设置各控件状态
        # ui.groupGrab.setEnabled(isOpen)
        # ui.groupParam.setEnabled(isOpen)

        self.bnOpen.setEnabled(not isOpen)
        self.bnClose.setEnabled(isOpen)

        self.bnStart.setEnabled(isOpen and (not isGrabbing))
        self.bnStop.setEnabled(isOpen and isGrabbing)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "DISPLAY"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Camera setting"))
        self.bnEnum.setText(_translate("MainWindow", "Enum"))
        self.bnOpen.setText(_translate("MainWindow", "Open"))
        self.bnClose.setText(_translate("MainWindow", "Close"))
        self.radioContinueMode.setText(_translate("MainWindow", "Continue"))
        self.radioTriggerMode.setText(_translate("MainWindow", "Trigger"))
        self.bnStart.setText(_translate("MainWindow", "Start"))
        self.label_2.setText(_translate("MainWindow", "Time trigger(ms):"))
        self.bnStop.setText(_translate("MainWindow", "Stop"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Param"))
        self.label_3.setText(_translate("MainWindow", "ExposureTime:"))
        self.label_4.setText(_translate("MainWindow", "Gain:"))
        self.label_5.setText(_translate("MainWindow", "FrameRate:"))
        self.bnGetParam.setText(_translate("MainWindow", "GetParam"))
        self.bnSetParam.setText(_translate("MainWindow", "SetParam"))
        self.Combo_model.setTitle(_translate("MainWindow", "Deep learning"))
        self.label_13.setText(_translate("MainWindow", "Model:"))
        self.ComboDevices_2.setItemText(0, _translate("MainWindow", "Solov2"))
        self.bnLoad_model.setText(_translate("MainWindow", "Load"))
        self.label_18.setText(_translate("MainWindow", "Time:(s): "))
        self.label_19.setText(_translate("MainWindow", "Score_Thr:"))
        self.bnRun.setText(_translate("MainWindow", "Run"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Calib Camera"))
        self.label_20.setText(_translate("MainWindow", "Load FIle: "))
        self.bnLoadCalib.setText(_translate("MainWindow", "Load"))
        self.groupBox_5.setTitle(_translate("MainWindow", "TCP/IP"))
        self.label_21.setText(_translate("MainWindow", "Send:"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    GUI1 = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(GUI1)
    GUI1.show()
    sys.exit(app.exec_())



