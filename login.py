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
        self.ui.pushButton_signup.clicked.connect(self.register_user)

    def check_login(self):
        username = self.ui.lineEdit_account.text()
        password = self.ui.lineEdit_password.text()

        user = self.database.fetchone("SELECT * FROM users WHERE username = %s", (username,))
        if user and user['password_hash'] == password:
            self.user_id = user['user_id']
            self.accept()
        else:
            QMessageBox.warning(self, "错误", "用户名或密码不正确！")
            
    def register_user(self):
        username = self.ui.lineEdit_account.text()
        password = self.ui.lineEdit_password.text()
        
        if not username or not password:
            QMessageBox.warning(self, "错误", "用户名和密码不能为空！")
            return

        existing_user = self.database.fetchone("SELECT * FROM users WHERE username = %s", (username,))
        if existing_user:
            QMessageBox.warning(self, "错误", "用户名已存在！")
            return
            
        try:
            self.database.execute("INSERT INTO users (username, password_hash, permission) VALUES (%s, %s, %s)", 
                                 (username, password, 'user'))
            QMessageBox.information(self, "成功", "注册成功！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"注册失败: {str(e)}")
