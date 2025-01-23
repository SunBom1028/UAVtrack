import cv2
import numpy as np
import os
import glob
from siamfc import TrackerSiamFC  # 确保正确导入SiamFC相关库


def select_roi(frame):
    # 用OpenCV选择ROI（感兴趣区域），返回选择的矩形
    bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")
    return bbox


if __name__ == '__main__':
    video_path = 'C:/Users/DELL/Documents/Overwatch/videos/overwatch/kaile_24-08-06_18-30-24.mp4'  # 替换为你的本地视频路径
    net_path = 'pretrained/siamfc_alexnet_e50.pth'  # 预训练模型路径

    # 初始化视频读取
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # 读取第一帧并选择ROI
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read video frame.")
        cap.release()
        exit()

    # 选择ROI
    x, y, w, h = select_roi(first_frame)
    init_bbox = [x, y, w, h]

    # 初始化SiamFC追踪器
    tracker = TrackerSiamFC(net_path=net_path)
    tracker.init(first_frame, init_bbox)

    # 逐帧进行追踪
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 更新跟踪器
        bbox = tracker.update(frame)
        x, y, w, h = map(int, bbox)

        # 绘制追踪框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow("Tracking", frame)

        # 按下Q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
