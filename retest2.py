import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QLabel, QComboBox, QLineEdit, QDialog, \
    QMessageBox, QTableWidget, QTableWidgetItem, QFrame
from siamfc import TrackerSiamFC
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QImage, QPixmap
import pymysql
import time
from denglu import Ui_myname
from trackui import Ui_MainWindow

class Database:
    def __init__(self, host, user, password, db):
        self.connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=db,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        self.cursor = self.connection.cursor()

    def execute(self, query, params=None):
        self.cursor.execute(query, params)
        self.connection.commit()

    def fetchone(self, query, params=None):
        self.cursor.execute(query, params)
        return self.cursor.fetchone()

    def fetchall(self, query, params=None):
        self.cursor.execute(query, params)
        return self.cursor.fetchall()

    def close(self):
        self.cursor.close()
        self.connection.close()

# 登录窗口
class LoginWindow(QDialog):
    def __init__(self, db):
        super().__init__()
        self.db = db
        self.user_id = None
        self.initUI()

    def initUI(self):
        self.ui = Ui_myname()
        self.ui.setupUi(self)
        self.ui.pushButton_access.clicked.connect(self.check_login)

    def check_login(self):
        username = self.ui.lineEdit_account.text()
        password = self.ui.lineEdit_password.text()

        user = self.db.fetchone("SELECT * FROM users WHERE username = %s", (username,))
        if user:
            # 校验密码（这里假设使用明文密码存储，实际应使用哈希密码）
            if user['password_hash'] == password:
                # 登录成功，检查用户是否有追踪记录表
                self.user_id = user['user_id']
                self.accept()
            else:
                QMessageBox.warning(self, "错误", "密码不正确！")
        else:
            QMessageBox.warning(self, "错误", "用户不存在！")

class VideoTracker(QMainWindow):
    def __init__(self, db, user_id, username):
        super().__init__()
        self.db = db
        self.user_id = user_id
        self.username = username
        self.tracking_table_name = None
        self.face_images_paths = {'left': None, 'front': None, 'right': None}
        self.is_tracking = False  # 添加标志

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.initUI()
        self.cap = None
        self.tracker = None
        self.bbox = None
        self.model_type = None
        self.face_image_path = None

        # 连接停止追踪按钮的信号
        self.ui.pushButton_object_stop.clicked.connect(self.stop_tracking)
        self.ui.pushButton_human_stop.clicked.connect(self.stop_tracking)
        self.ui.pushButton_multiface_stop.clicked.connect(self.stop_tracking)

        # 连接上传人脸图片按钮的信号
        self.ui.pushButton_faceimage_upload.clicked.connect(self.load_face_image)  # 单人脸上传
        self.ui.pushButton_multifaceimage_upload.clicked.connect(self.load_face_image)  # 多人脸上传

        # 连接查看历史记录按钮的信号
        self.ui.pushButton_history.clicked.connect(self.load_history)

    def stop_tracking(self):
        """停止追踪的方法"""
        self.is_tracking = False  # 设置标志为 False
        if self.cap is not None:
            self.cap.release()  # 释放视频捕获
        cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口

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

    def initUI(self):
        # 这里可以移除原有的 UI 初始化代码
        # 直接使用 trackui.py 中的 UI 组件
        self.ui.pushButton_object.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(0))
        self.ui.pushButton_human.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(1))
        self.ui.pushButton_multiface.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(2))
        self.ui.pushButton_history.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(3))

        # 连接上传视频和摄像头按钮
        self.ui.pushButton_object_upload.clicked.connect(self.load_video)  # 物体追踪上传视频
        self.ui.pushButton_human_upload.clicked.connect(self.load_video)  # 人类追踪上传视频
        self.ui.pushButton_multiface_upload.clicked.connect(self.load_video)  # 多角度人脸上传视频
        self.ui.pushButton_object_camera.clicked.connect(self.use_webcam)  # 物体追踪摄像头
        self.ui.pushButton_human_camera.clicked.connect(self.use_webcam)  # 人类追踪摄像头
        self.ui.pushButton_multiface_camera.clicked.connect(self.use_webcam)  # 多角度人脸摄像头

    def load_face_image(self):
            self.face_image_path, _ = QFileDialog.getOpenFileName(self, "Select Face Image", "",
                                                         "Image Files (*.jpg *.png *.jpeg)")
            if self.face_image_path:
                self.face_image = cv2.imread(self.face_image_path, cv2.IMREAD_COLOR)
                


    def load_video(self):
        # 打开文件对话框选择视频
        video_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi)")
        if video_path:
            if self.ui.stackedWidget.currentIndex() == 1 and self.face_image_path is None:  # 人类追踪
                QMessageBox.warning(self, "错误", "请先上传目标人脸图片")
            elif self.ui.stackedWidget.currentIndex() == 2:  # 多角度人脸
                self.multi_angle_facenet_track(video_path)
            else:  # 物体追踪
                self.siamfc_track(video_path)
    

    def select_roi(self, frame):
        # 用OpenCV选择ROI（感兴趣区域），返回选择的矩形
        bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")
        return bbox

    def update_label_with_frame(self, frame, is_human_tracking=False):
        """将 OpenCV 图像更新到 QLabel 中"""
        # 将 BGR 图像转换为 RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = frame_rgb.shape
        bytes_per_line = channels * width
        qimg = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # 根据是否是人类追踪选择 QLabel
        if is_human_tracking:
            self.ui.label_video_human.setPixmap(QPixmap.fromImage(qimg))
        else:
            self.ui.label_video_object.setPixmap(QPixmap.fromImage(qimg))

    def siamfc_track(self, video_path):
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

        QMessageBox.information(self, "信息", "目标已确定，开始追踪，按q退出")
        self.is_tracking = True  # 设置标志为 True

        # 初始化SiamFC追踪器
        self.tracker = TrackerSiamFC(net_path=net_path)
        self.tracker.init(first_frame, init_bbox)

        while self.is_tracking and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # 更新跟踪器
            bbox = self.tracker.update(frame)
            x, y, w, h = map(int, bbox)

            # 绘制追踪框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # 更新 QLabel 显示视频帧
            self.update_label_with_frame(frame)

            # 更新表格记录边界框
            self.update_table(x, y, w, h)

            #
            if self.is_tracking==False:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def facenet_track(self, video_path):
        self.start_new_tracking()

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('在该设备上运行: {}'.format(device))

        # 初始化人脸检测器和人脸识别模型
        mtcnn = MTCNN(keep_all=True, device=device)
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

        # 加载目标人脸图像并计算嵌入
        target_image_path = self.face_image_path
        target_image = Image.open(target_image_path)
        target_image_aligned, _ = mtcnn(target_image, return_prob=True)

        # 检查并调整目标人脸的格式
        if target_image_aligned is not None:
            target_image_aligned = target_image_aligned.to(device)
            target_embedding = resnet(target_image_aligned).detach().cpu()
            QMessageBox.information(self, "信息", "目标已确定，开始追踪，按q退出")
        else:
            raise ValueError("未检测到目标人脸，请检查目标图像")

        # 加载视频文件
        #video_path = 'video.mp4'
        video = cv2.VideoCapture(video_path)

        # 设置视频输出
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码器
        out = cv2.VideoWriter('tracked_video.mp4', fourcc, 25.0, (640, 360))

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            # 将帧转换为 PIL 图像格式并进行人脸检测
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            aligned_faces, probs = mtcnn(frame_pil, return_prob=True)

            if aligned_faces is not None:
                aligned_faces = aligned_faces.to(device)

                embeddings = resnet(aligned_faces).detach().cpu()

                # 计算目标嵌入与检测到的每张人脸之间的距离
                distances = [(target_embedding - emb).norm().item() for emb in embeddings]
                min_dist_index = np.argmin(distances)  # 找到最小距离的索引
                min_dist_box = mtcnn.detect(frame_pil)[0][min_dist_index]

                x, y, w, h = min_dist_box

                self.db.execute(f"""
                                INSERT INTO {self.tracking_table_name} (x, y, w, h)
                                VALUES (%s, %s, %s, %s)
                            """, (x, y, w, h))

                # 更新表格记录边界框
                self.update_table(x, y, w, h)

                # 在帧上绘制跟踪的边界框
                draw = ImageDraw.Draw(frame_pil)
                draw.rectangle(min_dist_box.tolist(), outline=(255, 0, 0), width=6)
                frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

            # 显示并保存当前帧
            cv2.imshow('Searching face and tracking', frame)
            out.write(cv2.resize(frame, (640, 360)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 释放资源
        video.release()
        out.release()
        cv2.destroyAllWindows()

    def update_table(self, x, y, w, h):
        # 根据当前选中的页面选择相应的表格
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

    def load_history(self):
        """加载历史记录并显示在 tableWidget_history 中"""
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

    def use_webcam(self): 
        # if self.comboBox.currentText() == "人类" and self.face_image_path is None:
       
        # 直接根据当前选中的页面调用相应的追踪方法
        if self.ui.stackedWidget.currentIndex() == 0:  # 物体追踪
            self.siamfc_track_webcam()
        elif self.ui.stackedWidget.currentIndex() == 1:  # 人类追踪
            self.facenet_track_webcam()
        elif self.ui.stackedWidget.currentIndex() == 2:  # 多角度人脸追踪
            self.multi_angle_facenet_track_webcam()

    def siamfc_track_webcam(self):
        self.start_new_tracking()

        net_path = 'pretrained/siamfc_alexnet_e50.pth'  # 预训练模型路径

        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("错误：无法打开摄像头")
            return

        # Read the first frame and select ROI
        ret, first_frame = cap.read()
        if not ret:
            print("错误：无法读取摄像头帧")
            cap.release()
            return

        # Select ROI
        time.sleep(1)
        x, y, w, h = self.select_roi(first_frame)
        init_bbox = [x, y, w, h]

        QMessageBox.information(self, "信息", "目标已确定，开始追踪，按q退出")

        # Initialize SiamFC tracker
        self.tracker = TrackerSiamFC(net_path=net_path)
        self.tracker.init(first_frame, init_bbox)

        while cap.isOpened() and self.is_tracking==True:
            ret, frame = cap.read()
            if not ret:
                break

            # Update tracker
            bbox = self.tracker.update(frame)
            x, y, w, h = map(int, bbox)

            self.db.execute(f"""
                            INSERT INTO {self.tracking_table_name} (x, y, w, h)
                            VALUES (%s, %s, %s, %s)
                        """, (x, y, w, h))

            # Draw tracking box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # 更新 QLabel 显示视频帧
            self.update_label_with_frame(frame)

            # 更新表格记录边界框
            self.update_table(x, y, w, h)

            # 按下Q退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def facenet_track_webcam(self):
        self.start_new_tracking()

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('在该设备上运行: {}'.format(device))

        # Initialize face detector and recognition model
        mtcnn = MTCNN(keep_all=True, device=device)
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

        # Load target face image and compute embedding
        target_image_path = self.face_image_path
        target_image = Image.open(target_image_path)
        target_image_aligned, _ = mtcnn(target_image, return_prob=True)

        if target_image_aligned is not None:
            target_image_aligned = target_image_aligned.to(device)
            target_embedding = resnet(target_image_aligned).detach().cpu()
            QMessageBox.information(self, "信息", "目标已确定，开始追踪，按q退出")
        else:
            raise ValueError("未检测到目标人脸，请检查目标图像")

        # Initialize webcam
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to PIL image and detect faces
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            aligned_faces, _ = mtcnn(frame_pil, return_prob=True)

            if aligned_faces is not None:
                aligned_faces = aligned_faces.to(device)
                detected_embeddings = resnet(aligned_faces).detach().cpu()

                # Calculate distances between target embedding and detected faces
                distances = [(target_embedding - emb).norm().item() for emb in detected_embeddings]
                min_dist_index = np.argmin(distances)
                min_dist_box = mtcnn.detect(frame_pil)[0][min_dist_index]

                x, y, w, h = min_dist_box

                self.db.execute(f"""
                                INSERT INTO {self.tracking_table_name} (x, y, w, h)
                                VALUES (%s, %s, %s, %s)
                            """, (x, y, w, h))

                # Update table with bounding box
                self.update_table(x, y, w, h)

                # Draw tracked bounding box on frame
                draw = ImageDraw.Draw(frame_pil)
                draw.rectangle(min_dist_box.tolist(), outline=(255, 0, 0), width=6)
                frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

                # Update QLabel with the current frame, indicating it's human tracking
                self.update_label_with_frame(frame, is_human_tracking=True)

            # Check for exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def multi_angle_facenet_track(self, video_path):
        self.start_new_tracking()

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        mtcnn = MTCNN(keep_all=True, device=device)
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

        # Load and compute embeddings for each angle
        embeddings = {}
        for angle, path in self.face_images_paths.items():
            if path:
                image = Image.open(path)
                aligned, _ = mtcnn(image, return_prob=True)
                if aligned is not None:
                    aligned = aligned.to(device)
                    embeddings[angle] = resnet(aligned).detach().cpu()

        if not embeddings:
            raise ValueError("未检测到任何目标人脸，请检查目标图像")

        video = cv2.VideoCapture(video_path)
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            aligned_faces, _ = mtcnn(frame_pil, return_prob=True)

            if aligned_faces is not None:
                aligned_faces = aligned_faces.to(device)
                detected_embeddings = resnet(aligned_faces).detach().cpu()

                # Calculate distances and apply weights
                weights = {'left': 0.3, 'front': 0.5, 'right': 0.2}
                min_dist = float('inf')
                min_dist_box = None

                for emb in detected_embeddings:
                    total_dist = sum(weights[angle] * (embeddings[angle] - emb).norm().item() for angle in embeddings)
                    if total_dist < min_dist:
                        min_dist = total_dist
                        min_dist_box = mtcnn.detect(frame_pil)[0][np.argmin(total_dist)]

                if min_dist_box is not None:
                    x, y, w, h = min_dist_box
                    self.db.execute(f"""
                                    INSERT INTO {self.tracking_table_name} (x, y, w, h)
                                    VALUES (%s, %s, %s, %s)
                                """, (x, y, w, h))
                    self.update_table(x, y, w, h)
                    draw = ImageDraw.Draw(frame_pil)
                    draw.rectangle(min_dist_box.tolist(), outline=(255, 0, 0), width=6)
                    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

            cv2.imshow('Multi-angle face tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

    def multi_angle_facenet_track_webcam(self):
        self.start_new_tracking()

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        mtcnn = MTCNN(keep_all=True, device=device)
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

        # Load and compute embeddings for each angle
        embeddings = {}
        for angle, path in self.face_images_paths.items():
            if path:
                image = Image.open(path)
                aligned, _ = mtcnn(image, return_prob=True)
                if aligned is not None:
                    aligned = aligned.to(device)
                    embeddings[angle] = resnet(aligned).detach().cpu()

        if not embeddings:
            raise ValueError("未检测到任何目标人脸，请检查目标图像")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.warning(self, "错误", "无法打开摄像头")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            aligned_faces, _ = mtcnn(frame_pil, return_prob=True)

            if aligned_faces is not None:
                aligned_faces = aligned_faces.to(device)
                detected_embeddings = resnet(aligned_faces).detach().cpu()

                # Calculate distances and apply weights
                weights = {'left': 0.3, 'front': 0.5, 'right': 0.2}
                min_dist = float('inf')
                min_dist_box = None

                for emb in detected_embeddings:
                    total_dist = sum(weights[angle] * (embeddings[angle] - emb).norm().item() for angle in embeddings)
                    if total_dist < min_dist:
                        min_dist = total_dist
                        min_dist_box = mtcnn.detect(frame_pil)[0][np.argmin(total_dist)]

                if min_dist_box is not None:
                    x, y, w, h = min_dist_box
                    self.db.execute(f"""
                                    INSERT INTO {self.tracking_table_name} (x, y, w, h)
                                    VALUES (%s, %s, %s, %s)
                                """, (x, y, w, h))
                    self.update_table(x, y, w, h)
                    draw = ImageDraw.Draw(frame_pil)
                    draw.rectangle(min_dist_box.tolist(), outline=(255, 0, 0), width=6)
                    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

            cv2.imshow('Multi-angle face tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    DB_HOST = 'localhost'
    DB_USER = 'root'
    DB_PASSWORD = '1234'  
    DB_NAME = 'tracker_db'
    db = Database(DB_HOST, DB_USER, DB_PASSWORD, DB_NAME)

    app = QApplication(sys.argv)

    # 显示登录窗口
    login_window = LoginWindow(db)
    if login_window.exec_() == QDialog.Accepted:
        user_id = login_window.user_id
        username = login_window.ui.lineEdit_account.text()
        
        main_window = VideoTracker(db, user_id, username)
        main_window.show()

        sys.exit(app.exec_())
        db.close() 
