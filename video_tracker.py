from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox, QTableWidgetItem, QInputDialog, QLineEdit
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
from list import Ui_Form  
from user_manager import UserManager 

import math

class VideoTracker(QMainWindow):
    def __init__(self, db, user_id, username):
        super().__init__()
        self.db = db
        self.user_id = user_id
        self.username = username
        self.tracking_table_name = None
        self.detail_windows = []  # 保存详情窗口

        self.is_tracking = False # 是否正在追踪的flag

        self.prev_position = None  # 上一帧目标位置
        self.prev_time = None      # 上一帧时间戳
        self.current_speed = 0     # 当前速度
        self.show_speed = True     # 是否显示速度

        self.timer = QTimer()  # 计时器
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
        self.left_face_path = None
        self.center_face_path = None
        self.right_face_path = None
        self.face_embeddings = None
        self.face_weights = None

        # 连接权重spinbox
        self.ui.doubleSpinBox_left.setValue(0.3)
        self.ui.doubleSpinBox_middle.setValue(0.4)
        self.ui.doubleSpinBox_right.setValue(0.3)
        
        # 连接权重改变信号
        self.ui.doubleSpinBox_left.valueChanged.connect(self.update_weights)
        self.ui.doubleSpinBox_middle.valueChanged.connect(self.update_weights)
        self.ui.doubleSpinBox_right.valueChanged.connect(self.update_weights)

        # 检查用户权限并设置用户管理按钮的可见性
        user = self.db.fetchone("SELECT permission FROM users WHERE user_id = %s", (self.user_id,))
        if user and user['permission'] == 'admin':
            self.ui.pushButton_usermanage.setVisible(True)
            self.ui.pushButton_usermanage.clicked.connect(self.show_user_manager)
        else:
            self.ui.pushButton_usermanage.setVisible(False)

        # 翻页&堆叠窗口
        self.ui.pushButton_object.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(0))
        self.ui.pushButton_human.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(1))
        self.ui.pushButton_multiface.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(2))
        self.ui.pushButton_history.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(3))

        # 连接物体追踪按钮
        self.ui.pushButton_object_upload.clicked.connect(self.object_video_track)  
        self.ui.pushButton_object_camera.clicked.connect(self.object_camera_track)  
        self.ui.pushButton_object_stop.clicked.connect(self.stop_tracking)

        # 连接人脸追踪按钮
        self.ui.pushButton_faceimage_upload.clicked.connect(self.load_face_image)  #单人脸上传
        self.ui.pushButton_human_upload.clicked.connect(self.human_video_track)  
        self.ui.pushButton_human_camera.clicked.connect(self.human_camera_track)  
        self.ui.pushButton_human_stop.clicked.connect(self.stop_tracking) 

        # 连接多角度人脸追踪按钮
        self.ui.pushButton_multifaceimage_upload.clicked.connect(self.load_multiface_images)  #多角度人脸上传
        self.ui.pushButton_multiface_upload.clicked.connect(self.multiface_video_track)
        self.ui.pushButton_multiface_camera.clicked.connect(self.multiface_camera_track)
        self.ui.pushButton_multiface_stop.clicked.connect(self.stop_tracking)

        # 连接查看历史记录按钮的信号
        self.ui.pushButton_history.clicked.connect(self.load_history)

        # 连接历史记录表格的点击事件
        self.ui.tableWidget_history.itemDoubleClicked.connect(self.show_tracking_detail)
        
        # 连接历史记录页面的新控件
        self.ui.comboBox.currentIndexChanged.connect(self.sort_history_records)
        self.ui.searchBox.textChanged.connect(self.filter_history_records)
        self.ui.deleteButton.clicked.connect(self.delete_tracking_record)

        # 速度显示开关按钮
        self.ui.checkBox_show_speed = QtWidgets.QCheckBox("显示速度")
        self.ui.checkBox_show_speed.setChecked(True)
        self.ui.checkBox_show_speed.stateChanged.connect(self.toggle_speed_display)
        self.ui.horizontalLayout.addWidget(self.ui.checkBox_show_speed)
        
    # 速度显示开关方法
    def toggle_speed_display(self, state):
        self.show_speed = state == 2  # QtCore.Qt.Checked = 2

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
                h INT NOT NULL,
                confidence FLOAT NOT NULL
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
        # 获取当前时间
        current_time = time.time()
        # 计算速度
        if self.prev_position is not None and self.prev_time is not None:
            # 计算位移（欧几里得距离）
            dx = x + w/2 - self.prev_position[0]  # 当前中心点x - 上一帧中心点x
            dy = y + h/2 - self.prev_position[1]  # 当前中心点y - 上一帧中心点y
            distance = math.sqrt(dx*dx + dy*dy)    # 像素位移
            # 计算时间差
            time_diff = current_time - self.prev_time  # 秒
            # 计算速度（像素/秒）
            if time_diff > 0:  # 避免除以零
                self.current_speed = distance / time_diff
        # 更新上一帧数据
        self.prev_position = (x + w/2, y + h/2)  # 保存中心点位置
        self.prev_time = current_time
        # 在绘制边界框后添加速度显示
        if self.show_speed:
            speed_text = f"速度: {self.current_speed:.1f} px/s"
            cv2.putText(frame, speed_text, (x, y - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        if not self.cap or not self.is_tracking:
            self.timer.stop()
            self.cap.release()
            cv2.destroyAllWindows()
            return

        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return

        if self.ui.stackedWidget.currentIndex() == 0:  # 物体追踪
            bbox = self.tracker.update(frame)
            x, y, w, h = map(int, bbox)
            
            # 获取响应图并计算置信度
            response = self.tracker.responses.squeeze().cpu().numpy()  # 获取响应图
            response = (response - response.min()) / (response.max() - response.min() + 1e-16)  # 归一化
            confidence = float(response.max())  # 使用响应图的最大值作为置信度

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # 更新显示
            qimg = self.update_label_with_frame(frame)
            self.ui.label_video_object.setPixmap(QPixmap.fromImage(qimg))

            # 保存追踪数据并更新表格
            self.db.execute(f"""
                INSERT INTO {self.tracking_table_name} (x, y, w, h, confidence)
                VALUES (%s, %s, %s, %s, %s)
            """, (x, y, w, h, confidence))
            self.update_table(x, y, w, h, confidence)

        elif self.ui.stackedWidget.currentIndex() == 1:  # 人类追踪
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            aligned_faces, probs = self.mtcnn(frame_pil, return_prob=True)

            current_frame = frame

            if aligned_faces is not None:
                aligned_faces = aligned_faces.to(self.device)
                detected_embeddings = self.resnet(aligned_faces).detach().cpu()

                # 计算距离并找到最匹配的人脸
                distances = [(self.target_embedding - emb).norm().item() for emb in detected_embeddings]
                min_dist_index = np.argmin(distances)
                min_dist = distances[min_dist_index]
                min_dist_box = self.mtcnn.detect(frame_pil)[0][min_dist_index]
                
                # 计算置信度 (距离越小，置信度越高)
                confidence = 1.0 / (1.0 + min_dist)

                x, y, w, h = min_dist_box

                draw = ImageDraw.Draw(frame_pil)
                draw.rectangle(min_dist_box.tolist(), outline=(255, 0, 0), width=6)
                current_frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

                self.db.execute(f"""
                    INSERT INTO {self.tracking_table_name} (x, y, w, h, confidence)
                    VALUES (%s, %s, %s, %s, %s)
                """, (x, y, w, h, confidence))
                self.update_table(x, y, w, h, confidence)

            qimg = self.update_label_with_frame(current_frame)
            self.ui.label_video_human.setPixmap(QPixmap.fromImage(qimg))

        elif self.ui.stackedWidget.currentIndex() == 2:  # 多角度人脸追踪
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            aligned_faces, probs = self.mtcnn(frame_pil, return_prob=True)

            current_frame = frame

            if aligned_faces is not None:
                aligned_faces = aligned_faces.to(self.device)
                detected_embeddings = self.resnet(aligned_faces).detach().cpu()
                
                weighted_distances = []
                for emb in detected_embeddings:
                    distances = [(target_emb - emb).norm().item() for target_emb in self.face_embeddings]
                    weighted_dist = sum(d * w for d, w in zip(distances, self.face_weights))
                    weighted_distances.append(weighted_dist)
                
                min_dist_index = np.argmin(weighted_distances)
                min_dist = weighted_distances[min_dist_index]
                min_dist_box = self.mtcnn.detect(frame_pil)[0][min_dist_index]
                
                # 计算置信度
                confidence = 1.0 / (1.0 + min_dist)
                
                x, y, w, h = min_dist_box
                
                draw = ImageDraw.Draw(frame_pil)
                draw.rectangle(min_dist_box.tolist(), outline=(255, 0, 0), width=6)
                current_frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
                
                self.db.execute(f"""
                    INSERT INTO {self.tracking_table_name} (x, y, w, h, confidence)
                    VALUES (%s, %s, %s, %s, %s)
                """, (x, y, w, h, confidence))
                self.update_table(x, y, w, h, confidence)

            qimg = self.update_label_with_frame(current_frame)
            self.ui.label_video_multiface.setPixmap(QPixmap.fromImage(qimg))

    def human_video_track(self):
        if self.face_image_path is None:
            QMessageBox.warning(self, "错误", "请先上传目标人脸图片")
            return

        video_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi)")
        self.start_new_tracking()

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
            
            # 确保表有confidence列
            self.ensure_table_has_confidence_column(table_name)
            
            self.ui.tableWidget_history.insertRow(row_count)
            self.ui.tableWidget_history.setItem(row_count, 0, QTableWidgetItem(table_name))
            self.ui.tableWidget_history.setItem(row_count, 1, QTableWidgetItem(str(created_at)))
            row_count += 1
            
        # 应用当前的排序和过滤设置
        self.sort_history_records()
        self.filter_history_records()
        
    def sort_history_records(self):
        """根据comboBox的选择对历史记录进行排序"""
        # 获取当前排序方式
        sort_order = self.ui.comboBox.currentIndex()  # 0为正序，1为倒序
        
        # 获取表格中的所有行
        rows = []
        for row in range(self.ui.tableWidget_history.rowCount()):
            table_name = self.ui.tableWidget_history.item(row, 0).text()
            created_at = self.ui.tableWidget_history.item(row, 1).text()
            rows.append((table_name, created_at))
        
        # 根据创建时间排序
        rows.sort(key=lambda x: x[1], reverse=(sort_order == 1))
        
        # 重新填充表格
        self.ui.tableWidget_history.setRowCount(0)
        for row, (table_name, created_at) in enumerate(rows):
            self.ui.tableWidget_history.insertRow(row)
            self.ui.tableWidget_history.setItem(row, 0, QTableWidgetItem(table_name))
            self.ui.tableWidget_history.setItem(row, 1, QTableWidgetItem(created_at))
            
    def filter_history_records(self):
        """根据searchBox的内容过滤历史记录"""
        search_text = self.ui.searchBox.text().lower()
        
        # 遍历所有行，隐藏不匹配的行
        for row in range(self.ui.tableWidget_history.rowCount()):
            table_name = self.ui.tableWidget_history.item(row, 0).text().lower()
            created_at = self.ui.tableWidget_history.item(row, 1).text().lower()
            
            # 如果搜索文本为空或表格名称/创建时间包含搜索文本，则显示该行
            if not search_text or search_text in table_name or search_text in created_at:
                self.ui.tableWidget_history.setRowHidden(row, False)
            else:
                self.ui.tableWidget_history.setRowHidden(row, True)
                
    def delete_tracking_record(self):
        """删除选中的追踪记录"""
        # 获取选中的行
        selected_rows = self.ui.tableWidget_history.selectedItems()
        if not selected_rows:
            QMessageBox.warning(self, "警告", "请先选择要删除的记录")
            return
            
        # 获取选中的行号（确保不重复）
        rows_to_delete = set()
        for item in selected_rows:
            rows_to_delete.add(item.row())
            
        # 确认删除
        reply = QMessageBox.question(self, "确认删除", 
                                    f"确定要删除选中的 {len(rows_to_delete)} 条记录吗？",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # 从后向前删除，避免索引变化
            for row in sorted(rows_to_delete, reverse=True):
                # 获取表名
                table_name = self.ui.tableWidget_history.item(row, 0).text()
                
                # 从数据库中删除记录
                self.db.execute("DELETE FROM tracking_tables WHERE table_name = %s", (table_name,))
                
                # 删除对应的追踪表
                self.db.execute(f"DROP TABLE IF EXISTS {table_name}")
                
                # 从表格中删除行
                self.ui.tableWidget_history.removeRow(row)
                
            QMessageBox.information(self, "成功", "记录已成功删除")

    def select_roi(self, frame):
        # 用OpenCV选择ROI（感兴趣区域），返回选择的矩形
        bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")
        return bbox
    
    def update_table(self, x, y, w, h, confidence):
        # 选择表格并更新数据
        row_position = 0
        if self.ui.stackedWidget.currentIndex() == 0:  # 物体追踪
            row_position = self.ui.tableWidget_object.rowCount()
            self.ui.tableWidget_object.insertRow(row_position)
            self.ui.tableWidget_object.setItem(row_position, 0, QTableWidgetItem(str(x)))
            self.ui.tableWidget_object.setItem(row_position, 1, QTableWidgetItem(str(y)))
            self.ui.tableWidget_object.setItem(row_position, 2, QTableWidgetItem(str(w)))
            self.ui.tableWidget_object.setItem(row_position, 3, QTableWidgetItem(str(h)))
            self.ui.tableWidget_object.setItem(row_position, 4, QTableWidgetItem(f"{confidence:.4f}"))
        elif self.ui.stackedWidget.currentIndex() == 1:  # 人类追踪
            row_position = self.ui.tableWidget_human.rowCount()
            self.ui.tableWidget_human.insertRow(row_position)
            self.ui.tableWidget_human.setItem(row_position, 0, QTableWidgetItem(str(x)))
            self.ui.tableWidget_human.setItem(row_position, 1, QTableWidgetItem(str(y)))
            self.ui.tableWidget_human.setItem(row_position, 2, QTableWidgetItem(str(w)))
            self.ui.tableWidget_human.setItem(row_position, 3, QTableWidgetItem(str(h)))
            self.ui.tableWidget_human.setItem(row_position, 4, QTableWidgetItem(f"{confidence:.4f}"))
        elif self.ui.stackedWidget.currentIndex() == 2:  # 多角度人脸追踪
            row_position = self.ui.tableWidget_multiface.rowCount()
            self.ui.tableWidget_multiface.insertRow(row_position)
            self.ui.tableWidget_multiface.setItem(row_position, 0, QTableWidgetItem(str(x)))
            self.ui.tableWidget_multiface.setItem(row_position, 1, QTableWidgetItem(str(y)))
            self.ui.tableWidget_multiface.setItem(row_position, 2, QTableWidgetItem(str(w)))
            self.ui.tableWidget_multiface.setItem(row_position, 3, QTableWidgetItem(str(h)))
            self.ui.tableWidget_multiface.setItem(row_position, 4, QTableWidgetItem(f"{confidence:.4f}"))

    def show_tracking_detail(self, item):
        row = item.row()
        table_name = self.ui.tableWidget_history.item(row, 0).text()
        
        # 确保表有confidence列
        self.ensure_table_has_confidence_column(table_name)
        
        # 创建并显示详情窗口
        detail_window = TrackingDetailWindow()
        detail_window.load_data(self.db, table_name)
        detail_window.show()
        
        # 将窗口保存到列表中
        self.detail_windows.append(detail_window)

    def show_user_manager(self):
        """显示用户管理界面"""
        user_manager = UserManager(self.db)
        user_manager.exec_()

    def load_multiface_images(self):
        """加载左中右三个角度的人脸图片"""
        self.left_face_path, _ = QFileDialog.getOpenFileName(self, "选择左侧人脸图片", "", "Image Files (*.jpg *.png *.jpeg)")
        if not self.left_face_path:
            return
            
        self.center_face_path, _ = QFileDialog.getOpenFileName(self, "选择正面人脸图片", "", "Image Files (*.jpg *.png *.jpeg)")
        if not self.center_face_path:
            return
            
        self.right_face_path, _ = QFileDialog.getOpenFileName(self, "选择右侧人脸图片", "", "Image Files (*.jpg *.png *.jpeg)")
        if not self.right_face_path:
            return

        # 初始化人脸检测和识别模型
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        # 计算三个人脸的嵌入向量
        self.face_embeddings = []
        face_paths = [self.left_face_path, self.center_face_path, self.right_face_path]
        
        for path in face_paths:
            img = Image.open(path)
            aligned_face, _ = self.mtcnn(img, return_prob=True)
            if aligned_face is not None:
                aligned_face = aligned_face.to(self.device)
                embedding = self.resnet(aligned_face).detach().cpu()
                self.face_embeddings.append(embedding)
            else:
                QMessageBox.warning(self, "错误", f"无法在图片中检测到人脸: {path}")
                return

        # 使用spinbox的值作为权重
        self.update_weights()

        QMessageBox.information(self, "成功", "已成功加载三个人脸图片")

    def multiface_video_track(self):
        """多角度人脸视频追踪"""
        if not all([self.left_face_path, self.center_face_path, self.right_face_path]):
            QMessageBox.warning(self, "错误", "请先上传三个人脸图片")
            return

        video_path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Video Files (*.mp4 *.avi)")
        if not video_path:
            return

        self.start_new_tracking()
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "错误", "无法打开视频文件")
            return

        self.timer.start(30)

    def multiface_camera_track(self):
        """多角度人脸摄像头追踪"""
        if not all([self.left_face_path, self.center_face_path, self.right_face_path]):
            QMessageBox.warning(self, "错误", "请先上传三个人脸图片")
            return

        self.start_new_tracking()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "错误", "无法打开摄像头")
            return

        self.timer.start(30)

    def update_weights(self):
        # 更新权重数组
        self.face_weights = [
            self.ui.doubleSpinBox_left.value(),
            self.ui.doubleSpinBox_middle.value(),
            self.ui.doubleSpinBox_right.value()
        ]

    def ensure_table_has_confidence_column(self, table_name):
        """确保表有confidence列，如果没有则添加"""
        try:
            # 检查表是否存在confidence列
            result = self.db.fetchone(f"""
                SELECT COUNT(*) as count 
                FROM information_schema.columns 
                WHERE table_name = '{table_name}' 
                AND column_name = 'confidence'
            """)
            
            # 如果没有confidence列，添加它
            if result['count'] == 0:
                self.db.execute(f"""
                    ALTER TABLE {table_name} 
                    ADD COLUMN confidence FLOAT NOT NULL DEFAULT 1.0
                """)
                print(f"Added confidence column to table {table_name}")
        except Exception as e:
            print(f"Error checking/adding confidence column: {str(e)}")

class TrackingDetailWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Form()  
        self.ui.setupUi(self)
        self.setWindowTitle("追踪记录详情")
        self.db = None
        self.table_name = None
        
        self.ui.renameButton.clicked.connect(self.rename_table)
    
    def load_data(self, db, table_name):
        # 保存数据库连接和表名
        self.db = db
        self.table_name = table_name
        
        
        # 设置表名作为标题
        self.ui.label_listname.setText(f"追踪记录: {table_name}")
        
        # 从数据库加载数据
        results = db.fetchall(f"SELECT x, y, w, h, confidence FROM {table_name}")
        
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
            self.ui.tableWidget_list.setItem(row, 4, QTableWidgetItem(f"{record['confidence']:.4f}"))
            
    def rename_table(self):
        """重命名当前追踪表"""
        if not self.db or not self.table_name:
            QMessageBox.warning(self, "错误", "无法重命名：未加载表数据")
            return
            
        # 弹出输入对话框获取新表名
        new_name, ok = QInputDialog.getText(
            self, 
            "重命名追踪表", 
            "请输入新表名:", 
            QLineEdit.Normal, 
            self.table_name
        )
        
        if ok and new_name and new_name != self.table_name:
            try:
                # 检查新表名是否已存在
                existing_tables = self.db.fetchall("SELECT table_name FROM tracking_tables")
                for table in existing_tables:
                    if table['table_name'] == new_name:
                        QMessageBox.warning(self, "错误", f"表名 '{new_name}' 已存在，请使用其他名称")
                        return
                
                # 重命名表
                self.db.execute(f"RENAME TABLE {self.table_name} TO {new_name}")
                
                # 更新tracking_tables表中的记录
                self.db.execute(
                    "UPDATE tracking_tables SET table_name = %s WHERE table_name = %s",
                    (new_name, self.table_name)
                )
                
                # 更新UI显示
                self.table_name = new_name
                self.ui.label_listname.setText(f"追踪记录: {new_name}")
                
                QMessageBox.information(self, "成功", f"表已成功重命名为 '{new_name}'")
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"重命名表时出错: {str(e)}")
