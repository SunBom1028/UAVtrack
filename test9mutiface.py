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
        self.setWindowTitle("登录界面")
        self.setGeometry(100, 100, 300, 200)

        self.username_label = QLabel("用户名：", self)
        self.username_label.move(20, 30)
        self.username_input = QLineEdit(self)
        self.username_input.move(100, 30)

        self.password_label = QLabel("密码：", self)
        self.password_label.move(20, 80)
        self.password_input = QLineEdit(self)
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.move(100, 80)

        self.login_btn = QPushButton("登录", self)
        self.login_btn.move(100, 130)
        self.login_btn.clicked.connect(self.check_login)

    def check_login(self):
        username = self.username_input.text()
        password = self.password_input.text()

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

class HistoryWindow(QDialog):
    def __init__(self, db, user_id):
        super().__init__()
        self.db = db
        self.user_id = user_id
        self.initUI()

    def initUI(self):
        self.setWindowTitle("历史追踪记录")
        self.setGeometry(200, 200, 300, 600)

        # 表格组件
        self.table_widget = QTableWidget(self)
        self.table_widget.setGeometry(20, 20, 260, 550)
        self.table_widget.setColumnCount(2)  # 列数：表名、时间、x, y, w, h
        self.table_widget.setHorizontalHeaderLabels(["表名", "创建时间"])

        self.load_history()

    def load_history(self):
        # 查询用户的所有追踪表
        tracking_tables = self.db.fetchall("""
            SELECT table_name, created_at FROM tracking_tables WHERE user_id = %s
        """, (self.user_id,))

        # 遍历每个追踪表，加载记录
        row_count = 0
        for table in tracking_tables:
            table_name = table["table_name"]
            created_at = table["created_at"]
            self.table_widget.insertRow(row_count)
            self.table_widget.setItem(row_count, 0, QTableWidgetItem(table_name))
            self.table_widget.setItem(row_count, 1, QTableWidgetItem(str(created_at)))
            row_count += 1


# 视频追踪主窗口
class VideoTracker(QMainWindow):
    def __init__(self,db,user_id,username):
        super().__init__()
        self.db = db
        self.user_id = user_id
        self.username = username
        self.tracking_table_name = None
        self.face_images_paths = {'left': None, 'front': None, 'right': None}

        self.initUI()
        self.cap = None
        self.tracker = None
        self.bbox = None
        self.model_type = None
        self.face_image_path = None

    def start_new_tracking(self):
        self.table_widget.clearContents()
        self.table_widget.setRowCount(0)
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
        font = QFont()
        font.setPointSize(12)

        # 选择目标类型下拉菜单
        self.comboBox = QComboBox(self)
        self.comboBox.addItem("物体")
        self.comboBox.addItem("人类")
        self.comboBox.addItem("多角度人脸")
        self.comboBox.move(50, 150)
        self.comboBox.currentIndexChanged.connect(self.on_target_type_changed)
        self.comboBox.setFont(font)

        self.btn_upload_face = QPushButton('上传面部图片', self)
        self.btn_upload_face.clicked.connect(self.load_face_image)
        self.btn_upload_face.move(50, 200)
        self.btn_upload_face.setFont(font)
        self.btn_upload_face.setVisible(False)  # 初始时隐藏按钮

        # 显示目标人脸图片
        self.face_image_label = QLabel(self)
        self.face_image_label.setGeometry(50, 350, 200, 200)
        self.face_image_label.setVisible(False)  # 初始时隐藏

        self.history_btn = QPushButton("查看历史记录", self)
        self.history_btn.move(50, 250)
        self.history_btn.clicked.connect(self.show_history)

        # 视频显示标签
        #self.label = QLabel(self)
        #self.label.setGeometry(50, 150, 640, 480)
        label1 = QLabel("追踪选项", self)
        label1.setGeometry(50, 50, 100, 50)
        label1.setStyleSheet("font-size: 20px; font-weight: bold;")

        label2 = QLabel("追踪记录", self)
        label2.setGeometry(300, 50, 100, 50)
        label2.setStyleSheet("font-size: 20px; font-weight: bold;")

        # 位置记录表格
        self.table_widget = QTableWidget(self)
        self.table_widget.setRowCount(0)
        self.table_widget.setColumnCount(4)
        self.table_widget.setHorizontalHeaderLabels(["X", "Y", "W", "H"])
        self.table_widget.setGeometry(300, 100, 400, 480)

        self.setGeometry(100, 100, 750, 630)
        self.setWindowTitle("追踪界面")

        v_line = QFrame(self)
        v_line.setGeometry(225, 0, 2, 1000)  # 设置位置(x, y)和宽度、高度
        v_line.setFrameShape(QFrame.VLine)  # 设置为垂直线
        v_line.setFrameShadow(QFrame.Sunken)  # 设置样式为凹陷

        self.btn_webcam = QPushButton('使用摄像头追踪', self)
        self.btn_webcam.clicked.connect(self.use_webcam)
        self.btn_webcam.move(50, 300)
        self.btn_webcam.setFont(font)

        # 选择视频按钮
        self.btn_open = QPushButton('上传视频追踪', self)
        self.btn_open.clicked.connect(self.load_video)
        self.btn_open.move(50, 350)
        self.btn_open.setFont(font)

    def on_target_type_changed(self):
        # 当目标类型选择发生变化时，如果选择的是Human，则显示上传人脸图片按钮
        if self.comboBox.currentText() == "人类":
            self.btn_upload_face.setVisible(True)  # 显示按钮
        elif self.comboBox.currentText() == "多角度人脸":
            self.btn_upload_face.setVisible(True)  # Show button for uploading face images
        else:
            self.btn_upload_face.setVisible(False)  # 隐藏按钮

    def load_face_image(self):
        if self.comboBox.currentText() == "多角度人脸":
            for angle in ['left', 'front', 'right']:
                path, _ = QFileDialog.getOpenFileName(self, f"Select {angle} Face Image", "",
                                                      "Image Files (*.jpg *.png *.jpeg)")
                if path:
                    self.face_images_paths[angle] = path
            QMessageBox.information(self, "信息", "多角度人脸图片加载成功！")
        else:
            self.face_image_path, _ = QFileDialog.getOpenFileName(self, "Select Face Image", "",
                                                         "Image Files (*.jpg *.png *.jpeg)")
            if self.face_image_path:
                self.face_image = cv2.imread(self.face_image_path, cv2.IMREAD_COLOR)
                if self.face_image is not None:
                    # 将图片转换为QPixmap显示在QLabel上
                    height, width, channels = self.face_image.shape
                    bytes_per_line = channels * width
                    qimg = QImage(self.face_image.data, width, height, bytes_per_line, QImage.Format_BGR888)
                    pixmap = QPixmap.fromImage(qimg)

                    # 设置图片为QLabel的内容
                    self.face_image_label.setPixmap(pixmap.scaled(100, 130))
                    self.face_image_label.setVisible(True)  # 显示QLabel
                    QMessageBox.information(self, "信息", "目标人脸图片加载成功！")
                else:
                    QMessageBox.warning(self, "错误", "目标人脸图片加载失败")


    def load_video(self):
        # 打开文件对话框选择视频
        video_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi)")
        if video_path:
            if self.comboBox.currentText() == "人类" and self.face_image_path is None:
                QMessageBox.warning(self, "错误", "请先上传目标人脸图片")
            elif self.comboBox.currentText() == "多角度人脸":
                self.multi_angle_facenet_track(video_path)
            else:
                if self.comboBox.currentText() == "物体":
                    self.siamfc_track(video_path)
                elif self.comboBox.currentText() == "人类":
                    self.facenet_track(video_path)

    

    def select_roi(self, frame):
        # 用OpenCV选择ROI（感兴趣区域），返回选择的矩形
        bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")
        return bbox

    def siamfc_track(self, video_path):
        self.start_new_tracking()

        net_path = 'pretrained/siamfc_alexnet_e50.pth'  # 预训练模型路径

        # 初始化视频读取
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("错误：无法打开视频")
            return

        # 读取第一帧并选择ROI
        ret, first_frame = cap.read()
        if not ret:
            print("错误：无法读取视频帧")
            cap.release()
            return

        # 选择ROI
        x, y, w, h = self.select_roi(first_frame)
        init_bbox = [x, y, w, h]

        QMessageBox.information(self, "信息", "目标已确定，开始追踪，按q退出")

        # 初始化SiamFC追踪器
        self.tracker = TrackerSiamFC(net_path=net_path)
        self.tracker.init(first_frame, init_bbox)

        # 打开文件写入追踪数据
        with open("tracking_data.txt", "w") as file:
            # 逐帧进行追踪
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 更新跟踪器
                bbox = self.tracker.update(frame)
                x, y, w, h = map(int, bbox)

                self.db.execute(f"""
                                INSERT INTO {self.tracking_table_name} (x, y, w, h)
                                VALUES (%s, %s, %s, %s)
                            """, (x, y, w, h))

                # 绘制追踪框
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.imshow('Tracking object', frame)

                # 记录追踪框位置到文件中
                file.write(f"{x},{y},{w},{h}\n")

                # 在表格中显示追踪数据
                self.update_table(x, y, w, h)

                # 按下Q退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
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
        # 新增一行并插入追踪数据
        row_position = self.table_widget.rowCount()
        self.table_widget.insertRow(row_position)
        self.table_widget.setItem(row_position, 0, QTableWidgetItem(str(x)))
        self.table_widget.setItem(row_position, 1, QTableWidgetItem(str(y)))
        self.table_widget.setItem(row_position, 2, QTableWidgetItem(str(w)))
        self.table_widget.setItem(row_position, 3, QTableWidgetItem(str(h)))

    def show_history(self):
        history_window = HistoryWindow(self.db, self.user_id)
        history_window.exec_()

    def use_webcam(self):
        if self.comboBox.currentText() == "人类" and self.face_image_path is None:
            QMessageBox.warning(self, "错误", "请先上传目标人脸图片")
        else:
            if self.comboBox.currentText() == "物体":
                self.siamfc_track_webcam()
            elif self.comboBox.currentText() == "人类":
                self.facenet_track_webcam()
            elif self.comboBox.currentText() == "多角度人脸":
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
        x, y, w, h = self.select_roi(first_frame)
        init_bbox = [x, y, w, h]

        QMessageBox.information(self, "信息", "目标已确定，开始追踪，按q退出")

        # Initialize SiamFC tracker
        self.tracker = TrackerSiamFC(net_path=net_path)
        self.tracker.init(first_frame, init_bbox)

        while cap.isOpened():
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
            cv2.imshow('Tracking object', frame)

            # Update table with tracking data
            self.update_table(x, y, w, h)

            # Press 'q' to exit
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
            aligned_faces, probs = mtcnn(frame_pil, return_prob=True)

            if aligned_faces is not None:
                aligned_faces = aligned_faces.to(device)

                embeddings = resnet(aligned_faces).detach().cpu()

                # Calculate distances between target embedding and detected faces
                distances = [(target_embedding - emb).norm().item() for emb in embeddings]
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

            # Display current frame
            cv2.imshow('Searching face and tracking', frame)
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
    DB_PASSWORD = '1234'  # 替换为你的数据库密码
    DB_NAME = 'tracker_db'
    db = Database(DB_HOST, DB_USER, DB_PASSWORD, DB_NAME)

    app = QApplication(sys.argv)

    # 显示登录窗口
    login_window = LoginWindow(db)
    if login_window.exec_() == QDialog.Accepted:
        user_id = login_window.user_id
        username = login_window.username_input.text()
        
        main_window = VideoTracker(db, user_id, username)
        main_window.show()

        sys.exit(app.exec_())
        db.close() 
