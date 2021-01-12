#coding:utf-8
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import Qt,QEvent
import sys
import cv2 as cv
import os
import numpy as np
from hair import getOrientation,get_PCA_angle,getWarpTile,curve_plot,get_width,auto_search
from status import Ui_Form
from pyqtgraph import PlotWidget
import pyqtgraph as pg
#全局变量，图像文件夹路径
imgPath = "imgs/"
#全局变量，结果存放变量
ptRslt = []
 
#图像标记类
class Mark(QMainWindow,Ui_Form):    
    def __init__(self):
        super(Mark, self).__init__()
        self.setupUi(self)
       # super(Mark,self).__init__()
        #左上角点100,100， 宽高1000,900， 可自己设置，未利用布局
        # self.setGeometry(100,100,1500,900)  
        # self.setWindowTitle("Mark")  #窗口标题
        self.initUI()
        self.image_origin = None
        self.image_ploted = None
        self.result=[]
        self.box_width_init = 20
        self.box_height_init = 60
        self.binary_threshold = 150
        self.img_loaded=False
        self.show_binary=False
        self.handle_index=-1
        
    def buttonClick(self):#label上显示文字hello world
        print('111111111')
    def initUI(self):
        pass

        # self.labelImg.grabKeyboard()   #控件开始捕获键盘

        self.buttonSave.clicked.connect(self.buttonClick)  #保存按钮关联的时间


        self.editBoxWidth.editingFinished.connect(self.boxWidthChange)
        self.editBoxHeight.editingFinished.connect(self.boxHeightChange)
     
        allImgs = os.listdir(imgPath)            #遍历路径，将所有文件放到列表框中
        for imgTmp in allImgs:
            self.allFiles.addItem(imgTmp)   # 将此文件添加到列表中
        self.allFiles.itemClicked.connect(self.itemClick)   #列表框关联时间，用信号槽的写法方式不起作用
 

        self.thresholdSlider.valueChanged.connect(self.thresholdUpdate)
        
        self.radioImageBinary.toggled.connect(self.binaryChecked)



        self.plot_widget = PlotWidget(self)
        self.plot_widget.setGeometry(QtCore.QRect(1230,450,200,200))
        self.width_count = np.zeros(30)
        self.x = np.arange(30)
        y = np.zeros(30)
        bg = pg.BarGraphItem(x=self.x, y=y, height=0, width=0.8)
        self.plot_widget.addItem(bg)
        #self.curve = self.plot_widget.plot(self.width_count, name="mode2")
        
 
    

    def mousePressEvent(self, QMouseEvent):      #鼠标单击事件
        self.handle_index = -1
        if self.img_loaded==False:
            return 
        fr = float(self.editR.text())          # 读取缩放比例
        pointT = QMouseEvent.pos()             # 获得鼠标点击处的坐标
        point = [int( (pointT.x()-200)/fr + 0.5),int((pointT.y()-70)/fr+0.5) ]

        # for result_ in self.result:
        #     rect = result_['rect']
        #     [[x,y],[w,h],angle] = rect

        #     dif = (x-point[0])*(x-point[0])+(y-point[1])*(y-point[1])
        #     if(dif<1000):
        #         print(dif)

        if(point[0]>0 and point[1]>0):
            if QMouseEvent.button()==Qt.LeftButton:
                img = self.image_origin.copy()
                pkg = auto_search(img,[point[0],point[1]], self.box_width_init, self.box_height_init,binary_threshold=self.binary_threshold,is_show=False)
                # 保存结果
                is_find = pkg['is_find']
                if(is_find):  
                    box = pkg['box']
                    rect = pkg['rect']
                    self.result.append({'rect':rect, 'box':box, 'width':rect[1][1]})

                self.image_ploted = img # 保存画上图案的图片
                self.imshow()
            elif QMouseEvent.button()==Qt.RightButton:
                min_distance = 99999
                min_idx = -1
                for idx,result in enumerate(self.result):
                    [[x,y],[w,h],angle] = result['rect']
                    dis = (point[0]-x)*(point[0]-x) + (point[1]-y)*(point[1]-y)
                    if(dis < min_distance):
                        min_distance = dis
                        min_idx = idx
                self.handle_index = min_idx
                self.imshow()
                pass

       
        self.width_count = np.zeros(30)
        for result_ in self.result:
            width = result_['width']
            if(width>30):   # 暂时认为不存在超过30宽度的毛发，后面改成自适应数组
                continue
            self.width_count[int(width)] += 1

        bg = pg.BarGraphItem(x=self.x, height=self.width_count, width=0.8)
        self.plot_widget.addItem(bg)

    def eventFilter(self,source, event):
        if event.type()==QEvent.MouseMove:
            [x,y] = [event.pos().x(),event.pos().y()]
            #print([x,y])
    
        return QMainWindow.eventFilter(self, source, event)
    

    def keyPressEvent(self, QKeyEvent):  # 键盘某个键被按下时调用
        if self.img_loaded==False:
            return
        #参数1  控件
        if(len(self.result)<1):
            return 
        rect=self.result[self.handle_index]['rect']
        [[x,y],[w,h],angle] = rect


        if QKeyEvent.key() == Qt.Key_A:  # 左移
            x -= 1
            rect = ((x,y),(w,h),angle) 
            box = cv.boxPoints(rect)
            box = np.int0(box)
            self.result.pop(self.handle_index)
            self.result.append({'rect':rect, 'box':box, 'width':rect[1][1]})
            self.handle_index = -1      # 修正后，对应目标的序号必定变为-1
            self.imshow()


        if QKeyEvent.key()== Qt.Key_D:  # 右移
            x += 1
            rect = ((x,y),(w,h),angle) 
            box = cv.boxPoints(rect)
            box = np.int0(box)
            self.result.pop(self.handle_index)      
            self.result.append({'rect':rect, 'box':box, 'width':rect[1][1]})
            self.handle_index = -1  
            self.imshow()


        if QKeyEvent.key()== Qt.Key_S:  # 下移
            y += 1
            rect = ((x,y),(w,h),angle) 
            box = cv.boxPoints(rect)
            box = np.int0(box)
            self.result.pop(self.handle_index)      
            self.result.append({'rect':rect, 'box':box, 'width':rect[1][1]})
            self.handle_index = -1  
            self.imshow()

        if QKeyEvent.key()== Qt.Key_W:  # 上移
            y -= 1
            rect = ((x,y),(w,h),angle) 
            box = cv.boxPoints(rect)
            box = np.int0(box)
            self.result.pop(self.handle_index)      
            self.result.append({'rect':rect, 'box':box, 'width':rect[1][1]})
            self.handle_index = -1  
            self.imshow()

        if QKeyEvent.key()== Qt.Key_Up:  # 变宽
            h += 1
            rect = ((x,y),(w,h),angle) 
            box = cv.boxPoints(rect)
            box = np.int0(box)
            self.result.pop(self.handle_index)      
            self.result.append({'rect':rect, 'box':box, 'width':rect[1][1]})
            self.handle_index = -1  
            self.imshow()

        if QKeyEvent.key()== Qt.Key_Down:   # 变窄
            h -= 1
            rect = ((x,y),(w,h),angle) 
            box = cv.boxPoints(rect)
            box = np.int0(box)
            self.result.pop(self.handle_index)      
            self.result.append({'rect':rect, 'box':box, 'width':rect[1][1]})
            self.handle_index = -1  
            self.imshow()

        if QKeyEvent.key()== Qt.Key_Left:   # 逆时针旋转
            angle -= 1
            rect = ((x,y),(w,h),angle) 
            box = cv.boxPoints(rect)
            box = np.int0(box)
            self.result.pop(self.handle_index)      
            self.result.append({'rect':rect, 'box':box, 'width':rect[1][1]})
            self.handle_index = -1  
            self.imshow()

        if QKeyEvent.key()== Qt.Key_Right:  # 顺时针旋转
            angle += 1
            rect = ((x,y),(w,h),angle) 
            box = cv.boxPoints(rect)
            box = np.int0(box)
            self.result.pop(self.handle_index)      
            self.result.append({'rect':rect, 'box':box, 'width':rect[1][1]})
            self.handle_index = -1  
            self.imshow()


        if QKeyEvent.key()== Qt.Key_Backspace:  # 删除
            self.result.pop(self.handle_index)
            self.handle_index = -1
            self.imshow()


        if QKeyEvent.modifiers() == Qt.ControlModifier|Qt.ShiftModifier and QKeyEvent.key() == Qt.Key_A:  # 三键组合
            print('按下了Ctrl+Shift+A键')

    def boxWidthChange(self):
        self.box_width_init = float(self.editBoxWidth.text())  
        self.editBoxWidth.clearFocus()
        return
        
    def boxHeightChange(self):
        self.box_height_init = float(self.editBoxHeight.text()) 
        self.editBoxHeight.clearFocus()
        return
         
    def imshow(self):
        try:
            if(self.show_binary==False):
                img = curve_plot(self.image_origin.copy(),self.result,(255,0,0),self.handle_index)
            else:
                img = cv.cvtColor(self.image_origin.copy(), cv.COLOR_BGR2GRAY)
                _, img = cv.threshold(img, self.binary_threshold, 255, cv.THRESH_BINARY_INV )
                img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
                img = curve_plot(img,self.result,(255,0,0),self.handle_index)
                # img = cv.cv
            img = cv.resize(img, self.img_size)
            img = QtGui.QImage(img, img.shape[1], img.shape[0],
                        img.shape[1]*3, QtGui.QImage.Format_RGB888)  # bytesPerLine参数设置为image的width*image.channels
            self.labelImg.setPixmap(QtGui.QPixmap(img))
        except: # 未加载图像
            pass

    def binaryChecked(self,isChecked):
        self.show_binary = isChecked
        self.imshow()
    def thresholdUpdate(self,value):
        self.binary_threshold = int(value)
        self.imshow()

    # 单机列表中某张图片，将其选中并显示
    def itemClick(self):  #列表框单击事件
        self.result=[]

        tmp = imgPath + self.allFiles.currentItem().text()  #图像的绝对路径
        print(tmp)
        src = cv.imread(str(tmp),1)      #读取图像
        src = cv.cvtColor(src, cv.COLOR_BGR2RGB) 

        self.image_origin = src.copy()
        self.image_ploted = src.copy()

        height = src.shape[0]             #图像高度
        ratioY = self.labelImg.height()/(height+0.0)  #按高度值缩放
        self.editR.setText(str(ratioY))    # 编辑框记录缩放值
 
        
        width = src.shape[1]           # 计算图像宽度，缩放图像
        height2 = self.labelImg.height()
        width2 = int(width*ratioY + 0.5)
        
        img2 = cv.resize(src, (width2, height2))

        self.img_size=(width2, height2)

        
        img2 = QtGui.QImage(img2, img2.shape[1], img2.shape[0],
                       img2.shape[1]*3, QtGui.QImage.Format_RGB888)  # bytesPerLine参数设置为image的width*image.channels
        self.labelImg.setPixmap(QtGui.QPixmap(img2))

        self.img_loaded=True
        self.allFiles.clearFocus()

 




if  __name__ == "__main__":                                 # main函数
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = Mark()
    MainWindow.show()
    app.installEventFilter(MainWindow)
    sys.exit(app.exec_())