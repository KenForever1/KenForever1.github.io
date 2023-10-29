1. 在使用resource加载qss文件后，如果改动了qss文件发现没有效果，直接使用qss文件路径，产生效果。是否改了qss文件后要重新生成resource.py文件。

```python
import resources

# file = QFile(":/newPrefix/Style/style.qss")
file = QFile("./Style/style.qss")
file.open(QFile.ReadOnly | QFile.Text)
stream = QTextStream(file)
stylesheet = stream.readAll()

app = QApplication(sys.argv)
app.setStyleSheet(stylesheet)
```
2.
QFrame默认有1px的边框，如果要取消可以采用下面的方式，但是这种方式会在Frame的子控件中应用，使得子控件也没有边框。
```
 self.frame.setStyleSheet("QFrame{ border: none; }")
```
通过#frame指定只取消QFrame的边框，不应用到子控件上。
```
 self.frame.setStyleSheet("QFrame#frame{ border: none; }")
```
3.
当设置了tableview的backgroud color和color后发现，左上角有一个地方不能设置，显得很突兀。
这个地方是Corner button，点击可以选中所有表格内容
要设置QTableCornerButton的背景色
```python
tableView.setStyleSheet(" QTableCornerButton::section { background-color: yellow;}")
```
4.
+ 
[how to change background color for qtabbar empty space pyqt](https://stackoverflow.com/questions/32100180/how-to-change-background-color-for-qtabbar-empty-space-pyqt)
As far as I know you can make it either via:

`tabWidget->setStyleSheet("QTabBar { background-color: red; }"); tabWidget->setDocumentMode(true);`

But it doesn't look good.

Or via:

`tabWidget->setAutoFillBackground(true); QPalette pal = tabWidget->palette(); pal.setColor(QPalette::Window, Qt::red); tabWidget->setPalette(pal);`

Or just create QWidget with some background and insert QTabWidget on top of it.

+ 
[PyQt5 Tabwidget tab bar blank area background color](https://stackoverflow.com/questions/60563477/pyqt5-tabwidget-tab-bar-blank-area-background-color)
It depends on the [`documentMode`](https://doc.qt.io/qt-5/qtabwidget.html#documentMode-prop).

If it's `False` (the default), you have to set the TabWidget background, and ensure that its [`autoFillBackground()`](https://doc.qt.io/qt-5/qwidget.html#autoFillBackground-prop) is set to True; this is very important, as Qt automatically unselect it when setting a stylesheet.  
Note that, in this case, the background will be "around" the whole tab widget too, if don't want it, just disable the border.

```python
    self.tabWidget.setStyleSheet('''
        QTabWidget {
            background: magenta;
            border: none;
        }
        QTabBar::tab {
            background: green;
        }
    ''')
```

If the document mode is on, instead, you can just set the background for the tab bar:

```python
    self.tabWidget.setStyleSheet('''
        QTabBar {
            background: magenta;
        }
        QTabBar::tab {
            background: green;
        }
    ''')
```
