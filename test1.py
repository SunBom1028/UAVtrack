import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QLabel, QComboBox, QLineEdit, QDialog, \
    QMessageBox, QTableWidget, QTableWidgetItem, QFrame
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from siamfc import TrackerSiamFC

# 登录窗口
class LoginWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("登录界面")
        self.setGeometry(100, 100, 300, 200)

        # 用户名输入框
        self.username_label = QLabel("用户：", self)
        self.username_label.move(20, 30)
        self.username_input = QLineEdit(self)
        self.username_input.move(100, 30)

        # 密码输入框
        self.password_label = QLabel("密码：", self)
        self.password_label.move(20, 80)
        self.password_input = QLineEdit(self)
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.move(100, 80)

        # 登录按钮
        self.login_btn = QPushButton("登录", self)
        self.login_btn.move(100, 130)
        self.login_btn.clicked.connect(self.check_login)

    def check_login(self):
        username = self.username_input.text()
        password = self.password_input.text()

        if username == "" and password == "":
            self.accept()
        else:
            QMessageBox.warning(self, "Error", "用户名或密码不正确！")


# 视频追踪主窗口
class VideoTracker(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.cap = None
        self.tracker = None
        self.bbox = None
        self.model_type = None

    def initUI(self):
        # 选择目标类型下拉菜单
        self.comboBox = QComboBox(self)
        self.comboBox.addItem("物体")
        self.comboBox.addItem("人类")
        self.comboBox.move(50, 100)

        # 选择视频按钮
        self.btn_open = QPushButton('打开视频', self)
        self.btn_open.clicked.connect(self.load_video)
        self.btn_open.move(50, 150)

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

    def load_video(self):
        # 打开文件对话框选择视频
        video_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi)")
        if video_path:
            self.siamfc_track(video_path)

    def select_roi(self, frame):
        # 用OpenCV选择ROI（感兴趣区域），返回选择的矩形
        bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")
        return bbox

    def siamfc_track(self, video_path):
        net_path = 'pretrained/siamfc_alexnet_e50.pth'  # 预训练模型路径

        # 初始化视频读取
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        # 读取第一帧并选择ROI
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Could not read video frame.")
            cap.release()
            return

        # 选择ROI
        x, y, w, h = self.select_roi(first_frame)
        init_bbox = [x, y, w, h]

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

                # 绘制追踪框
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.imshow("追踪中", frame)

                # 记录追踪框位置到文件中
                file.write(f"{x},{y},{w},{h}\n")

                # 在表格中显示追踪数据
                self.update_table(x, y, w, h)

                # 按下Q退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    def update_table(self, x, y, w, h):
        # 新增一行并插入追踪数据
        row_position = self.table_widget.rowCount()
        self.table_widget.insertRow(row_position)
        self.table_widget.setItem(row_position, 0, QTableWidgetItem(str(x)))
        self.table_widget.setItem(row_position, 1, QTableWidgetItem(str(y)))
        self.table_widget.setItem(row_position, 2, QTableWidgetItem(str(w)))
        self.table_widget.setItem(row_position, 3, QTableWidgetItem(str(h)))

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 显示登录窗口
    login_window = LoginWindow()
    if login_window.exec_() == QDialog.Accepted:
        # 登录成功后显示主窗口
        main_window = VideoTracker()
        main_window.show()
        sys.exit(app.exec_())
