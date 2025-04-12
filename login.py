from PyQt5.QtWidgets import QDialog, QMessageBox
from denglu import Ui_myname

class LoginWindow(QDialog):
    def __init__(self, database):
        super().__init__()
        self.database = database
        self.user_id = None
        self.initUI()

    def initUI(self):
        self.ui = Ui_myname()
        self.ui.setupUi(self)
        self.ui.pushButton_access.clicked.connect(self.check_login)

    def check_login(self):
        username = self.ui.lineEdit_account.text()
        password = self.ui.lineEdit_password.text()

        user = self.database.fetchone("SELECT * FROM users WHERE username = %s", (username,))
        if user and user['password_hash'] == password:
            self.user_id = user['user_id']
            self.accept()
        else:
            QMessageBox.warning(self, "错误", "用户名或密码不正确！")
