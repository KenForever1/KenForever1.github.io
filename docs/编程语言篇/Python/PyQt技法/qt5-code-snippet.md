## PyQt5常用代码片段

### 获取显示器分辨率
```python
        # 获取显示器分辨率
        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.screenheight = self.screenRect.height()
        self.screenwidth = self.screenRect.width()

        self.height = int(self.screenheight * 0.8)
        self.width = int(self.screenwidth * 0.8)

        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 3)
        self.splitter.setSizes([int(self.width * 0.5), int(self.width * 0.5)])

        print("Screen height {}".format(self.screenheight))
        print("Screen width {}".format(self.screenwidth))
```

### TCP网络

tcpServer例子：
```python
    HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
    PORT = 3238  # Port to listen on (non-privileged ports are > 1023)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        index = 0
        try:
            with conn:
                print(f"Connected by {addr}")
                while True:
                    data = ""
                    if index % 2 == 0:
                        data = getXxxData()
                    else:
                        data = getXxxData()
                    if not data:
                        break

                    dataLen = len(data)
                    print("send data len: ", dataLen)
                    header = dataLen.to_bytes(4, byteorder='little', signed=True)
                    sendData = header + data.encode()
                    conn.sendall(sendData)
                    time.sleep(1)
                    index += 1
        except KeyboardInterrupt:
            s.close()
            sys.exit()
```

tcpClient例子：
```python
class TcpClient(object):
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.host = "127.0.0.1"
        self.port = 3238
        self.dataQueue = deque()

    def connect(self, host: str = "127.0.0.1", port: int = 3238) -> (str, bool):
        self.host = host
        self.port = port
        # Connect to the remote host and port
        print("start sim TcpClient ... ")

        try:
            self.sock.connect((self.host, self.port))
        except Exception as e:
            print(e)
            return e.args[1], False
        return "", True

    def start(self):

        # send register info

        msg_type = '1'
        msg_len = 9
        id = 0
        reg_data = msg_type.encode() + \
                   msg_len.to_bytes(4, byteorder='little', signed=True) + \
                   id.to_bytes(4, byteorder='little', signed=True)
        self.sock.send(reg_data)

        ptl_header_len = 4  # 包头用 4 个字节长度表示
        alldata = b""
        while True:
            while True:
                if len(alldata) < ptl_header_len:  # 一直到包头字节流全部接收完成
                    alldata += self.sock.recv(4)
                    if not alldata:  # 收到服务器的套接字的 close 消息
                        self.sock.close()
                        return
                else:
                    header = int.from_bytes(alldata[:ptl_header_len], byteorder='little')  # 反序列化包头
                    body_len = header
                    print("recv header ：", body_len)
                    break

            while True:
                recv_all_len = ptl_header_len + body_len
                if len(alldata) < recv_all_len:  # 一直到包内容的字节流全部接收完成
                    alldata += self.sock.recv(recv_all_len - len(alldata))
                    continue
                else:
                    body_data = alldata[ptl_header_len: recv_all_len]
                    print("recv body len ：", len(body_data))

                    self.dataQueue.append(body_data)

                    alldata = b""
                    break

    def close(self):
        # Terminate
        self.sock.close()
```

### QListWidget

```python
def clearChosenItems(listWidget: QListWidget):
    for i in range(listWidget.count()):
        item = listWidget.item(i)
        item.setSelected(False)
```

```python
item = QListWidgetItem()
# if nodeState == "online":
#     item.setForeground(Qt.green)
# elif nodeState == "offline":
#     item.setForeground(Qt.red)
# item.setText("{} ({})".format(nodeName, nodeState))

item.setText(nodeName)
item.setData(QtCore.Qt.UserRole, nodeName)
self.listWidgetOfTaskGroupInfo.addItem(item)
```

### Json example
```python
import json
import jsonpickle
from json import JSONEncoder

class Employee(object):
    def __init__(self, name, salary, address):
        self.name = name
        self.salary = salary
        self.address = address

class Address(object):
    def __init__(self, city, street, pin):
        self.city = city
        self.street = street
        self.pin = pin

address = Address("Alpharetta", "7258 Spring Street", "30004")
employee = Employee("John", 9000, address)

print("Encode Object into JSON formatted Data using jsonpickle")
empJSON = jsonpickle.encode(employee, unpicklable=False)
print(type(empJSON))

print("Writing JSON Encode data into Python String")
employeeJSONData = json.dumps(empJSON, indent=4)
print(employeeJSONData)
print(type(employeeJSONData))

print("Decode JSON formatted Data using jsonpickle")
EmployeeJSON = jsonpickle.decode(employeeJSONData)
print(EmployeeJSON)
print(type(EmployeeJSON))

# Let's load it using the load method to check if we can decode it or not.
print("Load JSON using loads() method")
employeeJSON = json.loads(EmployeeJSON)
print(employeeJSON)
print(type(employeeJSON))
```

### PyQt5中异步刷新UI
```
PyQt5中异步刷新UI和Python中的多线程总结
https://blog.csdn.net/zcs_xueli/article/details/109209065
PyQt5界面刷新以及多线程更新UI数据实例
https://blog.csdn.net/hu694028833/article/details/80977302
```

```python
class Example(QThread):
    signal = pyqtSignal()    # 括号里填写信号传递的参数
    def __init__(self):
        super().__init__()

    def __del__(self):
        self.wait()

    def run(self):
        # 进行任务操作
        self.signal.emit()    # 发射信号

# UI类中
def buttonClick(self)
    self.thread = Example()
    self.thread.signal.connect(self.callback)
    self.thread.start()    # 启动线程

def callbakc(self):
    pass
```

```
Moving From one widget to another PyQt5
https://stackoverflow.com/questions/72461323/moving-from-one-widget-to-another-pyqt5
```