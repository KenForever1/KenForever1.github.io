### pyqt5程序打包成deb


参考：[packaging-pyqt5-applications-linux-pyinstaller](https://www.pythonguis.com/tutorials/packaging-pyqt5-applications-linux-pyinstaller/)



#### pyinstaller打包成可执行文件
```
pyinstaller --name myApp formMain.py
```
会生成一个myApp.spec 文件，可以在文件中修改打包配置

修改配置文件后，下次打包只需要执行如下命令：
```
pyinstaller myApp.spec
```

#### fpm打包成deb安装包

package.sh
```
#!/bin/sh
# Create folders.
[ -e package ] && rm -r package
mkdir -p package/opt
#mkdir -p package/usr/share/applications
# mkdir -p package/usr/share/icons/hicolor/scalable/apps

# Copy files (change icon names, add lines for non-scaled icons)
cp -r dist/myApp package/opt/myApp
# cp icons/penguin.svg package/usr/share/icons/hicolor/scalable/apps/hello-world.svg
# cp hello-world.desktop package/usr/share/applications

# Change permissions
find package/opt/myApp -type f -exec chmod 755 -- {} +
# find package/usr/share -type f -exec chmod 644 -- {} +
```

.fpm
```
-C package
-s dir
-t deb
-n "myApp"
-v 0.1.0
-p myApp.deb
```

#### 脚本执行

```shell
pyinstaller knowledgeShareApp.spec
./package.sh
fpm
```
