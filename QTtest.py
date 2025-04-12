import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QLineEdit

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = QWidget()
    w.setWindowTitle('QTtest')

    btn = QPushButton('caonima')
    btn.setParent(w)

    label = QLabel('caonimaB', w)
    label.setGeometry(10, 10, 100, 100)

    edit = QLineEdit(w)
    edit.setPlaceholderText('oi')
    edit.setGeometry(10, 20, 100, 100)

    w.resize(300, 300)

    w.show()
    app.exec_()