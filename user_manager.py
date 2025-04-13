from PyQt5.QtWidgets import QDialog, QMessageBox, QTableWidgetItem
from user import Ui_table

class UserManager(QDialog):
    def __init__(self, db):
        super().__init__()
        self.db = db
        self.initUI()
        
    def initUI(self):
        self.ui = Ui_table()
        self.ui.setupUi(self)
        
        # 连接信号和槽
        self.ui.comboBox_user.currentIndexChanged.connect(self.sort_users)
        self.ui.searchBox_user.textChanged.connect(self.filter_users)
        self.ui.deleteButton_user.clicked.connect(self.delete_user)
        
        # 加载用户数据
        self.load_users()
        
    def load_users(self):
        """加载所有用户数据到表格"""
        # 清空表格
        self.ui.tableWidget_user.clearContents()
        self.ui.tableWidget_user.setRowCount(0)
        
        # 从数据库获取所有用户
        users = self.db.fetchall("SELECT user_id, username, permission FROM users")
        
        # 填充表格
        for user in users:
            row = self.ui.tableWidget_user.rowCount()
            self.ui.tableWidget_user.insertRow(row)
            self.ui.tableWidget_user.setItem(row, 0, QTableWidgetItem(str(user['user_id'])))
            self.ui.tableWidget_user.setItem(row, 1, QTableWidgetItem(user['username']))
            self.ui.tableWidget_user.setItem(row, 2, QTableWidgetItem(user['permission']))
            
    def sort_users(self):
        """根据下拉框选择排序用户"""
        sort_order = self.ui.comboBox_user.currentIndex()  # 0为正序，1为倒序
        
        # 获取所有行
        rows = []
        for row in range(self.ui.tableWidget_user.rowCount()):
            user_id = int(self.ui.tableWidget_user.item(row, 0).text())
            username = self.ui.tableWidget_user.item(row, 1).text()
            permission = self.ui.tableWidget_user.item(row, 2).text()
            rows.append((user_id, username, permission))
            
        # 根据用户ID排序
        rows.sort(key=lambda x: x[0], reverse=(sort_order == 1))
        
        # 重新填充表格
        self.ui.tableWidget_user.setRowCount(0)
        for user_id, username, permission in rows:
            row = self.ui.tableWidget_user.rowCount()
            self.ui.tableWidget_user.insertRow(row)
            self.ui.tableWidget_user.setItem(row, 0, QTableWidgetItem(str(user_id)))
            self.ui.tableWidget_user.setItem(row, 1, QTableWidgetItem(username))
            self.ui.tableWidget_user.setItem(row, 2, QTableWidgetItem(permission))
            
    def filter_users(self):
        """根据搜索框内容过滤用户"""
        search_text = self.ui.searchBox_user.text().lower()
        
        # 遍历所有行，隐藏不匹配的行
        for row in range(self.ui.tableWidget_user.rowCount()):
            username = self.ui.tableWidget_user.item(row, 1).text().lower()
            permission = self.ui.tableWidget_user.item(row, 2).text().lower()
            
            # 如果搜索文本为空或用户名/权限包含搜索文本，则显示该行
            if not search_text or search_text in username or search_text in permission:
                self.ui.tableWidget_user.setRowHidden(row, False)
            else:
                self.ui.tableWidget_user.setRowHidden(row, True)
                
    def delete_user(self):
        """删除选中的用户"""
        # 获取选中的行
        selected_items = self.ui.tableWidget_user.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选择要删除的用户")
            return
            
        # 获取选中的行号（确保不重复）
        rows_to_delete = set()
        for item in selected_items:
            rows_to_delete.add(item.row())
            
        # 确认删除
        reply = QMessageBox.question(self, "确认删除", 
                                   f"确定要删除选中的 {len(rows_to_delete)} 个用户吗？",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # 从后向前删除，避免索引变化
            for row in sorted(rows_to_delete, reverse=True):
                user_id = int(self.ui.tableWidget_user.item(row, 0).text())
                permission = self.ui.tableWidget_user.item(row, 2).text()
                
                # 检查是否为管理员
                if permission == 'admin':
                    QMessageBox.warning(self, "警告", "不能删除管理员账号！")
                    continue
                    
                try:
                    # 从数据库中删除用户
                    self.db.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
                    
                    # 从表格中删除行
                    self.ui.tableWidget_user.removeRow(row)
                    
                except Exception as e:
                    QMessageBox.critical(self, "错误", f"删除用户时出错: {str(e)}")
                    
            QMessageBox.information(self, "成功", "用户已成功删除") 