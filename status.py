# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\status.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1500, 821)
        self.labelImg = QtWidgets.QLabel(Form)
        self.labelImg.setGeometry(QtCore.QRect(200, 70, 1000, 600))
        self.labelImg.setText("")
        self.labelImg.setObjectName("labelImg")
        self.buttonSave = QtWidgets.QPushButton(Form)
        self.buttonSave.setGeometry(QtCore.QRect(400, 20, 75, 23))
        self.buttonSave.setObjectName("buttonSave")
        self.allFiles = QtWidgets.QListWidget(Form)
        self.allFiles.setGeometry(QtCore.QRect(10, 40, 171, 711))
        self.allFiles.setObjectName("allFiles")
        self.editR = QtWidgets.QLineEdit(Form)
        self.editR.setGeometry(QtCore.QRect(279, 17, 71, 20))
        self.editR.setObjectName("editR")
        self.editBoxHeight = QtWidgets.QLineEdit(Form)
        self.editBoxHeight.setGeometry(QtCore.QRect(1290, 200, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.editBoxHeight.setFont(font)
        self.editBoxHeight.setText("")
        self.editBoxHeight.setObjectName("editBoxHeight")
        self.editBoxWidth = QtWidgets.QLineEdit(Form)
        self.editBoxWidth.setGeometry(QtCore.QRect(1290, 150, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.editBoxWidth.setFont(font)
        self.editBoxWidth.setText("")
        self.editBoxWidth.setObjectName("editBoxWidth")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(1240, 150, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(1240, 200, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_2.setFont(font)
        self.label_2.setLineWidth(1)
        self.label_2.setWordWrap(False)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setGeometry(QtCore.QRect(1200, 90, 251, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Form)
        self.label_4.setGeometry(QtCore.QRect(870, 670, 161, 131))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_6 = QtWidgets.QLabel(Form)
        self.label_6.setGeometry(QtCore.QRect(212, 19, 71, 16))
        self.label_6.setObjectName("label_6")
        self.thresholdSlider = QtWidgets.QSlider(Form)
        self.thresholdSlider.setGeometry(QtCore.QRect(1220, 330, 160, 22))
        self.thresholdSlider.setFocusPolicy(QtCore.Qt.NoFocus)
        self.thresholdSlider.setMaximum(255)
        self.thresholdSlider.setPageStep(1)
        self.thresholdSlider.setProperty("value", 150)
        self.thresholdSlider.setOrientation(QtCore.Qt.Horizontal)
        self.thresholdSlider.setObjectName("thresholdSlider")
        self.label_7 = QtWidgets.QLabel(Form)
        self.label_7.setGeometry(QtCore.QRect(1220, 280, 161, 31))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(Form)
        self.label_8.setGeometry(QtCore.QRect(1390, 140, 61, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.radioImageBinary = QtWidgets.QRadioButton(Form)
        self.radioImageBinary.setGeometry(QtCore.QRect(658, 15, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.radioImageBinary.setFont(font)
        self.radioImageBinary.setFocusPolicy(QtCore.Qt.NoFocus)
        self.radioImageBinary.setObjectName("radioImageBinary")
        self.radioImageOrigin = QtWidgets.QRadioButton(Form)
        self.radioImageOrigin.setGeometry(QtCore.QRect(568, 15, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.radioImageOrigin.setFont(font)
        self.radioImageOrigin.setFocusPolicy(QtCore.Qt.NoFocus)
        self.radioImageOrigin.setChecked(True)
        self.radioImageOrigin.setObjectName("radioImageOrigin")
        self.label_9 = QtWidgets.QLabel(Form)
        self.label_9.setGeometry(QtCore.QRect(440, 670, 281, 131))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(Form)
        self.label_10.setGeometry(QtCore.QRect(1210, 390, 251, 51))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.buttonSave.setText(_translate("Form", "buttonOpen"))
        self.editR.setText(_translate("Form", "1.0"))
        self.editBoxHeight.setPlaceholderText(_translate("Form", "60.0"))
        self.editBoxWidth.setPlaceholderText(_translate("Form", "20.0"))
        self.label.setText(_translate("Form", "宽度"))
        self.label_2.setText(_translate("Form", "长度"))
        self.label_3.setText(_translate("Form", "搜索框初始尺寸设置"))
        self.label_4.setText(_translate("Form", "平移：WASD\n"
"旋转：←→\n"
"调宽：↑↓"))
        self.label_6.setText(_translate("Form", "图片缩放比"))
        self.label_7.setText(_translate("Form", "二值化阈值"))
        self.label_8.setText(_translate("Form", "有bug"))
        self.radioImageBinary.setText(_translate("Form", "二值图"))
        self.radioImageOrigin.setText(_translate("Form", "原图"))
        self.label_9.setText(_translate("Form", "生成：鼠标左键\n"
"选中：鼠标右键\n"
"删除：回退键"))
        self.label_10.setText(_translate("Form", "统计图：数量-发宽"))
