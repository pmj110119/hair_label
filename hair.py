import cv2
import numpy as np

import matplotlib.pyplot as plt
from math import *

def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    
def getOrientation(pts, img):    
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    

    if(len(eigenvectors)<2):
        return 1000
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    #     drawAxis(img, cntr, p1, (0, 255, 0), 1) #绿色，较长轴
    #     drawAxis(img, cntr, p2, (255, 255, 0), 1) #黄色
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians #PCA第一维度的角度    
    return angle



def get_PCA_angle(src,binary_threshold=150):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, binary_threshold, 255, cv2.THRESH_BINARY_INV )
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    areas=[]
    areas_angle=[]
    for i, c in enumerate(contours):
        # 面积筛选,暂时不用    后面用百分比来滤
        area = cv2.contourArea(c)
        if area < 10:
           # print('面积太小：',area)
            continue
        rect = cv2.minAreaRect(c)
        [w,h] = rect[1]
        if(w>h):
            temp = w
            w = h
            h = temp
        if(h/w<2.5):
            continue
        #cv2.drawContours(src, contours, i, (0, 0, 255), 2)
        #plt.figure(1);plt.imshow(src,cmap='gray')
        angle=getOrientation(c, src)    # Find the orientation of each shape
        if(angle==1000):
            continue
        areas.append(area)
        areas_angle.append(angle)
    
    #plt.figure(2);plt.imshow(src,cmap='gray')
    if(len(areas)==0):
        return {'angle':None, 'binary':None}
    ind=np.argmax(areas)
 
    return {'angle':areas_angle[ind]*57.3, 'binary':bw}



def getWarpTile(img,center,rect_width,rect_height,rect_angle, inverse=False):
    [xp_Click,yp_Click] = center
    ysize,xsize = img.shape[:2]
    # 计算四个角点坐标
    p1 = [round(max(0,xp_Click-rect_height//2)),round(max(0,yp_Click-rect_width//2))]
    p2 = [round(max(0,xp_Click-rect_height//2)),round(min(ysize,yp_Click+rect_width//2))]
    p3 = [round(min(xsize,xp_Click+rect_height//2)),round(min(ysize,yp_Click+rect_width//2))]
    p4 = [round(min(xsize,xp_Click+rect_height//2)),round(max(0,yp_Click-rect_width//2))]
    xg = np.array([p1[0],p2[0],p3[0],p4[0]])
    yg = np.array([p1[1],p2[1],p3[1],p4[1]])
    xg_t = xp_Click + (xg-xp_Click)*cos(rect_angle) + (yg-yp_Click)*sin(rect_angle) # 旋转
    yg_t = yp_Click + (yg-yp_Click)*cos(rect_angle) - (xg-xp_Click)*sin(rect_angle)
    cnt = np.array([
        [[int(xg_t[0]), int(yg_t[0])]],
        [[int(xg_t[1]), int(yg_t[1])]],
        [[int(xg_t[2]), int(yg_t[2])]],
        [[int(xg_t[3]), int(yg_t[3])]]
    ])
    
    # 得到旋转后的box
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    width = int(rect[1][0])
    height = int(rect[1][1])
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    tile = cv2.warpPerspective(img, M, (width, height))
    
    # 90度校正——不加的话会有图片横竖不一的bug
    if(inverse==False):
        if(tile.shape[0]>tile.shape[1]):
            tile = cv2.rotate(tile, cv2.ROTATE_90_CLOCKWISE)
            [[cX,cY],[w,h],angle] = rect
            rect = ((cX,cY),(h,w),angle+90)
    else:
        if(tile.shape[0]<tile.shape[1]):
            tile = cv2.rotate(tile, cv2.ROTATE_90_CLOCKWISE)
            [[cX,cY],[w,h],angle] = rect
            rect = ((cX,cY),(h,w),angle+90)

    return {'tile':tile,'rect':rect, 'box':box}

def get_width(tile,threshold=150):

    tile_gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    _, tile_binary = cv2.threshold(tile_gray, threshold, 255, cv2.THRESH_BINARY_INV )
    contours, _ = cv2.findContours(tile_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    area_max = 0
    cnt = []
    for i, c in enumerate(contours):  # 只考虑最大轮廓
        area = cv2.contourArea(c)
        if(area>area_max):
            area_max = area
            cnt = c
    if(len(cnt)==0):
        return [0,0] 
    else:
        x, y, w, h = cv2.boundingRect(cnt)         #用一个矩形将轮廓包围
        return [h,(y+h/2)-tile.shape[0]/2.0]


def auto_search(src,center,rect_width,rect_height,binary_threshold=150,is_show=True):
    img = src.copy()
    ## 1.角度校正
    FOUND_ANGLE = False
    rotate_angle_corrected = None
    visual_height = None
    for rotate_angle in range(0,180):
        result = getWarpTile(img,center, rect_width, rect_height, rect_angle=rotate_angle/57.3)
        
        tile = result['tile']
        rect = result['rect']
   
        visual_height = rect[1][1]
        box = result['box']
        # 计算主成分角度
        PCA_result = get_PCA_angle(tile,binary_threshold)
        angle = PCA_result['angle']
        tile_binary = PCA_result['binary']
        
        if(is_show):
            temp_img = img.copy()
            cv2.drawContours(temp_img, [box], 0, (0, 0, 255), 2)
            cv2.imshow('image',temp_img)
            key=cv2.waitKey(1)
   

        if(angle!=None):
            # cv2.imshow('tile_binary',tile_binary)
            # key=cv2.waitKey(1)
   
            if(angle<1 and angle>-1 ):
                FOUND_ANGLE = True
                rotate_angle_corrected = rotate_angle
                rect_rotate_corrected = rect
                break
    if(FOUND_ANGLE==False):
        return {'is_find':False, 'box':None}
    
  
    ## 2.中心偏移校正    (不需要卡的很死，只要保证毛发整体进入视野就行)
    Y_diff = 100
    new_center = center
    niter = 0
    while(abs(Y_diff)>1.0):
        # 计算当前box最大轮廓的质心坐标               # 这里30是为了缩短宽度，减少影响，但不能比width更短，不然rect会反90度（因为warp时强行设置width>height了，后面修正）
        result = getWarpTile(img,new_center, rect_width, 30, rect_angle=rotate_angle_corrected/57.3)
        tile = result['tile']
        rect = result['rect']


        rect = (rect[0],(rect[1][0],visual_height),rect_rotate_corrected[2])

        rect_origin = rect

        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # 计算偏移量（Y_diff）
        tile_gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        _, tile_binary = cv2.threshold(tile_gray, binary_threshold, 255, cv2.THRESH_BINARY_INV )
        contours, _ = cv2.findContours(tile_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        area_max = 0
        cY = None
        for i, c in enumerate(contours):  # 只考虑最大轮廓
            area = cv2.contourArea(c)
            if(area>area_max):
                area_max = area
                rect = cv2.minAreaRect(c)
                [[cX,cY],[w,h],angle] = rect
                up = cY + h/2
                down = cY - h/2
        if(cY == None):
            return {'is_find':False, 'box':None}
        # 根据偏移量的正负进行微调
        Y_diff = cY - rect_width/2
        if(Y_diff<0):# box向上挪一个单位
            new_center[1]-=0.5
        else:        # box向下挪一个单位
            new_center[1]+=0.5
        
        # 异常
        niter += 1
        if(niter>20):
            return {'is_find':False, 'box':None}
  
    ## 3.宽度校正    原本20x70    改为20x5，用这一小截来确定宽度
    result = getWarpTile(img,new_center, 20, 5, rect_angle=rotate_angle_corrected/57.3,inverse=True)        
    tile = result['tile']
    # 测宽
    [height,y_offset] = get_width(tile,binary_threshold)
   # print(height,y_offset)
    if(height > 0):
        [xx,yy]=rect_origin[0]
        rect = ((xx,yy+y_offset),(rect_rotate_corrected[1][0],height),rect_origin[2])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        return {'is_find':True, 'rect':rect, 'box':box}
    else:
        return {'is_find':False, 'rect':None, 'box':None}

    


# 绘制

def curve_plot(img,results,color=(0, 0, 255),handle_index=None):
    curve = img.copy()
    width = []
    for result in results:
        box = result['box']
        width.append(result['width'])
        cv2.drawContours(curve, [box], 0, color, 2)
    if(handle_index!=None and len(results)>0):
        box = results[handle_index]['box']
        [r,g,b]=color
        cv2.drawContours(curve, [box], 0, (g,r,b), 2)
    return curve

    



if __name__ == "__main__":

    img = cv2.imread('imgs/2.jpg')
    curve = img.copy()
    result=[]
    
    # 鼠标双击的回调函数
    def action(event,x_click,y_click,flags,param):
        global result
        if event == cv2.EVENT_LBUTTONDBLCLK:
            # process
            pkg = auto_search(curve,[x_click,y_click],30,60)
            # 保存结果
            is_find = pkg['is_find']
            if(is_find):  
                box = pkg['box']
                rect = pkg['rect']
                #print('rect:',rect)
                result.append({'rect':rect, 'box':box, 'width':rect[1][1]})

    cv2.namedWindow('image')
    cv2.setMouseCallback('image',action)
    while(True):
        curve = curve_plot(img,result)
        cv2.imshow('image',curve)
        command = cv2.waitKey(1)
        if(command==27):
            cv2.destroyAllWindows()
            break
        elif(command==100): # 'd'
            result.pop(-1)
            print(len(result))
        elif(command==115): # 's'
            print(len(result))
            width = []
            for result_ in result:
                width.append(result_['width'])
            plt.hist(np.array(width))
            plt.figure(1)
            plt.show()
        elif(command!=-1):
            print(command)

        

        