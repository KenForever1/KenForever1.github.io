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

```python
if __name__=="__main__":
    import sys
    app = QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    sys.exit(app.exec_())
```
