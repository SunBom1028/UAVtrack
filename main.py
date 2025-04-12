import sys
from PyQt5.QtWidgets import QApplication, QDialog
from database import Database
from login import LoginWindow
from video_tracker import VideoTracker

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 数据库连接信息
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': '1234',
        'db': 'tracker_db'
    }
    database = Database(**db_config)

    # 显示登录窗口
    login_window = LoginWindow(database)
    if login_window.exec_() == QDialog.Accepted:
        user_id = login_window.user_id
        username = login_window.ui.lineEdit_account.text()
        
        main_window = VideoTracker(database, user_id, username)
        main_window.show()

        sys.exit(app.exec_())

    database.close()
