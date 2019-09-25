import cv2
from yes_or_no import istrue

# 读取视频
camera = cv2.VideoCapture('./mouse003_m.mp4')

# 背景建模
history = 25
bs = cv2.createBackgroundSubtractorKNN(detectShadows=False)
bs.setHistory(history)

frames = 0
while True:
    res, frame = camera.read()

    if not res:
        break

    # 根据history参数设定前多少帧进行建模
    fg_mask = bs.apply(frame)   # 获取 foreground mask

    if frames < history:
        frames += 1
        continue

    # 二值化和膨胀腐蚀
    th = cv2.threshold(fg_mask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
    th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
    dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)

    # 得到目标轮廓
    _ , contours, hier = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:

        #第一个过滤条件：面积大于25像素小于2500的判定为老鼠
        if 25 < cv2.contourArea(c) < 2500:
            # 获取矩形框边界坐标
            (x, y, w, h) = cv2.boundingRect(c)

            # 扩大边界框
            y_start=y-25
            y_end=y+h+25
            x_start=x-25
            x_end=x+w+25
            cut_image=frame[y_start:y_end,x_start:x_end]
            
            # 第二个过滤条件，使用二分类判断目标是否是老鼠
            bo=istrue(cut_image)
            if bo:
                # 将目标画出来
                cv2.rectangle(frame, (x_start, y_end), (x_end,y_start),(0,255,0), 2)

    # 显示最终图像
    cv2.namedWindow('video',cv2.WINDOW_GUI_NORMAL)
    cv2.imshow('video',frame)
    if cv2.waitKey(110) & 0xff == 27:
         break
