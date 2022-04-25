### 安装配置环境

#### 虚拟环境配置
使用pycharm创建项目，选择venv选项，自动在项目目录中创建python venv环境，目前使用python 3.8版本。

首先创建一个pyQt5Space的pyqt5项目工作空间，虚拟环境在此目录下创建，后面项目都在pyQt5Space目录下使用eric6进行创建，生成代码。

##### 安装开发环境
requirements.txt
```
msgpack==1.0.2
pyqt5==5.15.2
pyqt5-tools==5.15.2.3.2
QScintilla==2.11.6
PyQtWebEngine==5.15.2
PyQtChart==5.15.2
pyqt5-plugins==5.15.2.2.2
eric-ide==20.12.1
```

```shell
pip install -r requirements.txt -i https://pypi.douban.com/simple
```
#### eric6

注意事项：
1. 在创建项目的时候，不能把项目地址和工作空间地址填成一样，否则OK按钮成灰色，无法创建项目。
2. 在关闭eric6后，重新打开项目，发现ui文件不再了，可以通过鼠标右键，add forms重新将项目中的ui文件添加进来，然后使用qtdesigner打开。
3. 使用qtdesigner设计好界面后，使用eric6添加事件函数，会自动生成代码文件，前后端分离。

#### 在生成的入口py文件中添加如下启动代码

```python
if __name__=="__main__":
    import sys
    app = QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    sys.exit(app.exec_())
```

参考资源：[pyqt5+eric6实战入门到精通（懒人的最爱，代码自动生成，你信吗？）](https://www.bilibili.com/video/BV1L54y157P1?spm_id_from=333.337.search-card.all.click)
