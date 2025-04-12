from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox, QTableWidgetItem
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from PyQt5 import QtWidgets

from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import time

from siamfc import TrackerSiamFC

from trackui import Ui_MainWindow
from list import Ui_Form  # 导入list.py中的UI类

class VideoTracker(QMainWindow):
    def __init__(self, db, user_id, username):
        super().__init__()
        self.db = db
        self.user_id = user_id
        self.username = username
        self.tracking_table_name = None
        self.detail_windows = []  # 添加一个列表来保存详情窗口

        self.is_tracking = False # flag

        self.timer = QTimer()  # 添加计时器
        self.timer.timeout.connect(self.update_frame)  # 连接更新帧的槽函数

        self.initUI()

    def initUI(self):
        # UI 初始化代码
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.cap = None
        self.tracker = None
        self.bbox = None
        self.model_type = None
        self.face_image_path = None

        # 翻页&堆叠窗口
        self.ui.pushButton_object.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(0))
        self.ui.pushButton_human.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(1))
        self.ui.pushButton_multiface.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(2))
        self.ui.pushButton_history.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(3))

        # 连接上传视频和摄像头按钮
        self.ui.pushButton_object_upload.clicked.connect(self.object_video_track)  # 物体追踪上传视频
        self.ui.pushButton_object_camera.clicked.connect(self.object_camera_track)  # 物体追踪摄像头
        self.ui.pushButton_object_stop.clicked.connect(self.stop_tracking) # 物体停止追踪

        self.ui.pushButton_faceimage_upload.clicked.connect(self.load_face_image)  # 单人脸上传
        self.ui.pushButton_human_upload.clicked.connect(self.human_video_track)  # 人类追踪上传视频
        self.ui.pushButton_human_camera.clicked.connect(self.human_camera_track)  # 人类追踪摄像头
        self.ui.pushButton_human_stop.clicked.connect(self.stop_tracking) #人类停止追踪

        #self.ui.pushButton_multifaceimage_upload.clicked.connect(self.load_face_image)  # 多人脸上传
        #self.ui.pushButton_multiface_upload.clicked.connect(self.load_video)  # 多角度人脸上传视频
        #self.ui.pushButton_multiface_camera.clicked.connect(self.use_webcam)  # 多角度人脸摄像头
        #self.ui.pushButton_multiface_stop.clicked.connect(self.stop_tracking) # 多角度停止追踪

        # 连接查看历史记录按钮的信号
        self.ui.pushButton_history.clicked.connect(self.load_history)

        # 连接历史记录表格的点击事件
        self.ui.tableWidget_history.itemDoubleClicked.connect(self.show_tracking_detail)

    def start_new_tracking(self):
        self.is_tracking=True
        # 根据当前选中的页面选择相应的表格
        if self.ui.stackedWidget.currentIndex() == 0:  # 物体追踪
            self.ui.tableWidget_object.clearContents()
            self.ui.tableWidget_object.setRowCount(0)
        elif self.ui.stackedWidget.currentIndex() == 1:  # 人类追踪
            self.ui.tableWidget_human.clearContents()
            self.ui.tableWidget_human.setRowCount(0)
        elif self.ui.stackedWidget.currentIndex() == 2:  # 多角度人脸追踪
            self.ui.tableWidget_multiface.clearContents()
            self.ui.tableWidget_multiface.setRowCount(0)

        # 生成唯一表名
        timestamp = int(time.time())  # 当前时间戳
        self.tracking_table_name = f"{self.username}_tracking_{timestamp}"

        # 创建新的追踪表
        self.db.execute(f"""
            CREATE TABLE {self.tracking_table_name} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                x INT NOT NULL,
                y INT NOT NULL,
                w INT NOT NULL,
                h INT NOT NULL
            );
        """)

        # 在 tracking_tables 中记录新表
        self.db.execute("""
            INSERT INTO tracking_tables (user_id, table_name) VALUES (%s, %s)
        """, (self.user_id, self.tracking_table_name))

    def stop_tracking(self):
        self.is_tracking = False 
        if hasattr(self, 'timer'):
            self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.is_tracking = False 
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

    def load_face_image(self):
        self.face_image_path, _ = QFileDialog.getOpenFileName(self, "Select Face Image", "", "Image Files (*.jpg *.png *.jpeg)")
        if self.face_image_path:
            self.face_image = cv2.imread(self.face_image_path, cv2.IMREAD_COLOR)

    def update_label_with_frame(self, frame):
        # 更新 QLabel 显示视频帧
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = frame_rgb.shape
        bytes_per_line = channels * width
        qimg = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return qimg
    
    def object_video_track(self):
        # 物体视频追踪
        video_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi)")

        self.start_new_tracking()
        net_path = 'pretrained/siamfc_alexnet_e50.pth'  # 预训练模型路径

        # 初始化视频读取
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print("错误：无法打开视频")
            return

        # 读取第一帧并选择ROI
        ret, first_frame = self.cap.read()
        if not ret:
            print("错误：无法读取视频帧")
            self.cap.release()
            return

        # 选择ROI
        x, y, w, h = self.select_roi(first_frame)
        init_bbox = [x, y, w, h]

        QMessageBox.information(self, "信息", "目标已确定，开始追踪")

        # 初始化SiamFC追踪器
        self.tracker = TrackerSiamFC(net_path=net_path)
        self.tracker.init(first_frame, init_bbox)
        
        # 启动定时器
        self.timer.start(30)

    def object_camera_track(self):
        # 物体摄像头追踪
        self.start_new_tracking()

        net_path = 'pretrained/siamfc_alexnet_e50.pth'  # 预训练模型路径

        # Initialize webcam
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("错误：无法打开摄像头")
            return

        # Read the first frame and select ROI
        ret, first_frame = self.cap.read()
        if not ret:
            print("错误：无法读取摄像头帧")
            self.cap.release()
            return

        # Select ROI
        x, y, w, h = self.select_roi(first_frame)
        init_bbox = [x, y, w, h]

        QMessageBox.information(self, "信息", "目标已确定，开始追踪")

        # Initialize SiamFC tracker
        self.tracker = TrackerSiamFC(net_path=net_path)
        self.tracker.init(first_frame, init_bbox)
        
        # 启动定时器，每30ms更新一次（约33fps）
        self.timer.start(30)

    def update_frame(self):
        if not self.cap or not self.is_tracking:
            self.timer.stop()
            self.cap.release()
            cv2.destroyAllWindows()
            return

        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return

        # 根据当前页面选择不同的追踪逻辑
        if self.ui.stackedWidget.currentIndex() == 0:  # 物体追踪
            # Update tracker
            bbox = self.tracker.update(frame)
            x, y, w, h = map(int, bbox)

            # tracking box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # 更新显示
            qimg = self.update_label_with_frame(frame)
            self.ui.label_video_object.setPixmap(QPixmap.fromImage(qimg))

            # 保存追踪数据并更新表格
            self.db.execute(f"""
                INSERT INTO {self.tracking_table_name} (x, y, w, h)
                VALUES (%s, %s, %s, %s)
            """, (x, y, w, h))
            self.update_table(x, y, w, h)

        else:  # 人脸追踪（包括视频和摄像头）
            # 转换为PIL图像并检测人脸
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            aligned_faces, _ = self.mtcnn(frame_pil, return_prob=True)

            # 无论是否检测到人脸，都显示当前帧
            current_frame = frame

            if aligned_faces is not None:
                # 确保张量在正确的设备上
                aligned_faces = aligned_faces.to(self.device)
                detected_embeddings = self.resnet(aligned_faces).detach().cpu()

                # 计算距离并找到最匹配的人脸
                distances = [(self.target_embedding - emb).norm().item() for emb in detected_embeddings]
                min_dist_index = np.argmin(distances)
                min_dist_box = self.mtcnn.detect(frame_pil)[0][min_dist_index]

                x, y, w, h = min_dist_box

                # 绘制边界框
                draw = ImageDraw.Draw(frame_pil)
                draw.rectangle(min_dist_box.tolist(), outline=(255, 0, 0), width=6)
                current_frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

                # 保存追踪数据并更新表格
                self.db.execute(f"""
                    INSERT INTO {self.tracking_table_name} (x, y, w, h)
                    VALUES (%s, %s, %s, %s)
                """, (x, y, w, h))
                self.update_table(x, y, w, h)

            # 更新显示
            qimg = self.update_label_with_frame(current_frame)
            self.ui.label_video_human.setPixmap(QPixmap.fromImage(qimg))

    def human_video_track(self):
        if self.face_image_path is None:
            QMessageBox.warning(self, "错误", "请先上传目标人脸图片")
            return

        video_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi)")
        self.start_new_tracking()

        # 设置设备并保存为类属性
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('在该设备上运行: {}'.format(self.device))

        # 初始化人脸检测器和人脸识别模型
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        # 加载目标人脸图像并计算嵌入
        target_image_path = self.face_image_path
        target_image = Image.open(target_image_path)
        target_image_aligned, _ = self.mtcnn(target_image, return_prob=True)

        # 检查并调整目标人脸的格式
        if target_image_aligned is not None:
            target_image_aligned = target_image_aligned.to(self.device)
            self.target_embedding = self.resnet(target_image_aligned).detach().cpu()
            QMessageBox.information(self, "信息", "目标已确定，开始追踪")
        else:
            raise ValueError("未检测到目标人脸，请检查目标图像")

        # 初始化视频捕获
        self.cap = cv2.VideoCapture(video_path)
        
        # 启动定时器
        self.timer.start(30)

    def human_camera_track(self):
        if self.face_image_path is None:
            QMessageBox.warning(self, "错误", "请先上传目标人脸图片")
            return
        
        self.start_new_tracking()

        # 设置设备并保存为类属性
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('在该设备上运行: {}'.format(self.device))

        # 初始化人脸检测器和识别模型
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        # 加载目标人脸图像并计算嵌入
        target_image_path = self.face_image_path
        target_image = Image.open(target_image_path)
        target_image_aligned, _ = self.mtcnn(target_image, return_prob=True)

        if target_image_aligned is not None:
            target_image_aligned = target_image_aligned.to(self.device)
            self.target_embedding = self.resnet(target_image_aligned).detach().cpu()
            QMessageBox.information(self, "信息", "目标已确定，开始追踪")
        else:
            raise ValueError("未检测到目标人脸，请检查目标图像")

        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        
        # 启动定时器
        self.timer.start(30)

    def load_history(self):
        # 清空历史记录表
        self.ui.tableWidget_history.clearContents()
        self.ui.tableWidget_history.setRowCount(0)

        # 查询用户的所有追踪表
        tracking_tables = self.db.fetchall("""
            SELECT table_name, created_at FROM tracking_tables WHERE user_id = %s
        """, (self.user_id,))

        # 遍历每个追踪表，加载记录
        row_count = 0
        for table in tracking_tables:
            table_name = table["table_name"]
            created_at = table["created_at"]
            self.ui.tableWidget_history.insertRow(row_count)
            self.ui.tableWidget_history.setItem(row_count, 0, QTableWidgetItem(table_name))
            self.ui.tableWidget_history.setItem(row_count, 1, QTableWidgetItem(str(created_at)))
            row_count += 1

    def select_roi(self, frame):
        # 用OpenCV选择ROI（感兴趣区域），返回选择的矩形
        bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")
        return bbox
    
    def update_table(self, x, y, w, h):
        # 选择表格并更新数据
        row_position = 0
        if self.ui.stackedWidget.currentIndex() == 0:  # 物体追踪
            row_position = self.ui.tableWidget_object.rowCount()
            self.ui.tableWidget_object.insertRow(row_position)
            self.ui.tableWidget_object.setItem(row_position, 0, QTableWidgetItem(str(x)))
            self.ui.tableWidget_object.setItem(row_position, 1, QTableWidgetItem(str(y)))
            self.ui.tableWidget_object.setItem(row_position, 2, QTableWidgetItem(str(w)))
            self.ui.tableWidget_object.setItem(row_position, 3, QTableWidgetItem(str(h)))
        elif self.ui.stackedWidget.currentIndex() == 1:  # 人类追踪
            row_position = self.ui.tableWidget_human.rowCount()
            self.ui.tableWidget_human.insertRow(row_position)
            self.ui.tableWidget_human.setItem(row_position, 0, QTableWidgetItem(str(x)))
            self.ui.tableWidget_human.setItem(row_position, 1, QTableWidgetItem(str(y)))
            self.ui.tableWidget_human.setItem(row_position, 2, QTableWidgetItem(str(w)))
            self.ui.tableWidget_human.setItem(row_position, 3, QTableWidgetItem(str(h)))
        elif self.ui.stackedWidget.currentIndex() == 2:  # 多角度人脸追踪
            row_position = self.ui.tableWidget_multiface.rowCount()
            self.ui.tableWidget_multiface.insertRow(row_position)
            self.ui.tableWidget_multiface.setItem(row_position, 0, QTableWidgetItem(str(x)))
            self.ui.tableWidget_multiface.setItem(row_position, 1, QTableWidgetItem(str(y)))
            self.ui.tableWidget_multiface.setItem(row_position, 2, QTableWidgetItem(str(w)))
            self.ui.tableWidget_multiface.setItem(row_position, 3, QTableWidgetItem(str(h)))

    def show_tracking_detail(self, item):
        # 获取被点击的行
        row = item.row()
        # 获取表名（在第一列）
        table_name = self.ui.tableWidget_history.item(row, 0).text()
        
        # 创建并显示详情窗口
        detail_window = TrackingDetailWindow()
        detail_window.load_data(self.db, table_name)
        detail_window.show()
        
        # 将窗口保存到列表中，防止被垃圾回收
        self.detail_windows.append(detail_window)

class TrackingDetailWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Form()  # 使用list.py中的UI
        self.ui.setupUi(self)
        self.setWindowTitle("追踪记录详情")
    
    def load_data(self, db, table_name):
        # 设置表名作为标题
        self.ui.label_listname.setText(f"追踪记录: {table_name}")
        
        # 从数据库加载数据
        results = db.fetchall(f"SELECT x, y, w, h FROM {table_name}")
        
        # 清空表格
        self.ui.tableWidget_list.clearContents()
        self.ui.tableWidget_list.setRowCount(0)
        
        # 填充数据
        for record in results:
            row = self.ui.tableWidget_list.rowCount()
            self.ui.tableWidget_list.insertRow(row)
            self.ui.tableWidget_list.setItem(row, 0, QTableWidgetItem(str(record['x'])))
            self.ui.tableWidget_list.setItem(row, 1, QTableWidgetItem(str(record['y'])))
            self.ui.tableWidget_list.setItem(row, 2, QTableWidgetItem(str(record['w'])))
            self.ui.tableWidget_list.setItem(row, 3, QTableWidgetItem(str(record['h'])))