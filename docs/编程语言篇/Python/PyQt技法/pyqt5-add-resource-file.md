### pyqt5中添加资源文件

最简单的情况，当需要向pyqt5中的控件中加入image的时候，需要将image以resource的方式导入pyqt5项目中。

首先在项目目录中创建一个Resources目录，作为资源文件的保存Dir，将资源文件放入这一目录。

通过Qt-designer，添加资源文件。

保存后，会生成一个imagesResource.qrc的文件。

使用如下命令，将imagesResource.qrc文件编译成py文件，通过eric6编译项目，会自动导入py文件的文件名进行使用。

```
pyrcc5 -o icon_rc.py icon.qrc
```

