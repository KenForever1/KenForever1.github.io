
```css
QGroupBox {
    border-radius: 10px;
	border-width: 5px;
	border-style: solid;
	border-color: #023656;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    top: 5px;
    padding: 0 5px;
    background-color: white;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}
```

[border-style](https://www.w3.org/TR/css-backgrounds-3/#border-style)

```css
/*
QLabel {
    color: red;
    font-size: 20px;
}
*/

/*
qlabel fit font size:
set sizePolicy Expanding
set wordWrap true
*/
QLabel#label {
    color: rgb(160, 210, 255);
    background-color: transparent;
}

QLabel#label_2 {
    color: rgb(77, 202, 210);
    background-color: transparent;
}

QLabel#label_3 {
    color: rgb(160, 210, 255);
    background-color: transparent;
}

QPushButton {
    /*灰底白字
    background-color: rgb(29,37,43);
    color: rgb(255,255,255);*/
    /*蓝底白字*/
    background-color: rgb(0,82,158);
    color: rgb(255,255,255);
    font-size: 16px;
}

/*
QGroupBox {
    border: 2px solid gray;
    border-radius: 10px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 0 5px;
    border-radius: 5px;
}

QGroupBox::corner {
    background-color: white;
    border: none;
    border-radius: 5px;
    width: 5px;
    height: 5px;
}
*/

QGroupBox {
    border-radius: 10px;
	border-width: 5px;
	border-style: solid;
	border-color: #023656;
    font-size: 18px;
}
QGroupBox::corner {
    background-color: red;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    top: 5px;
    color: #A0D2FF;
    background-color: #002D4D;
    border-top-right-radius: 8px;
}

/*
#groupBox2 {
    background-color: rgb(0, 21, 40);
    font-size: 16px;
    font-weight: bold;
    color: red;
}
*/

#centralWidget {
    background-color: rgb(0, 21, 40);
}


```

```python
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QFile, QTextStream

file = QFile("Style/style.qss")
file.open(QFile.ReadOnly | QFile.Text)
stream = QTextStream(file)
stylesheet = stream.readAll()

app = QApplication(sys.argv)
app.setStyleSheet(stylesheet)
main_window = MainWindow()
main_window.show()
sys.exit(app.exec_())
```


[qtcreator 7.0.2](https://download.qt.io/official_releases/qtcreator/7.0/7.0.2/installer_source/linux_x64/)

[qml application](https://www.pythonguis.com/tutorials/qml-qtquick-python-application/)

[非梦教程](https://www.zhihu.com/people/fei-meng-38-58/posts)

[cross-platform-gui-application-based-on-pyqt](https://leovan.me/cn/2018/05/cross-platform-gui-application-based-on-pyqt/)

[qmlintegration](https://doc.qt.io/qtforpython/tutorials/qmlintegration/qmlintegration.html)
