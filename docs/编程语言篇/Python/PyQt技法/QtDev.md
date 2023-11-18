### data notify

+ 数据通知使用pyqtSignal，pyQtSignal声明类需要是QtCore.QObject的子类，否则会报错
  pyqtSignal可以通过函数构造函数传参的方式，在不同的QObject类或者QWidget类中传递，信号就可以在不同类中传递通知

```python
class xxxClass(QtCore.QObject):
    xxxSignal: pyqtSignal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.xxxSignal.connect(self.xxxSignalCallback)

    def xxx_emit(self):
        self.xxxSignal.emit("xxxx")

    def xxxSignalCallback(self, info):
        print("recv signal : " + info)
```

+ 在PyQt中，创建线程使用QThread，不要使用Threading库， 否则会卡住UI线程

```python

```

### parse bytes data

struct.unpack

struct.pack

### 数据的存储和数据的显示分离

https://www.pythonguis.com/tutorials/qtableview-modelviews-numpy-pandas/

#### data model

数据通过网络等接受后，保持到List或DataTable package等数据结构中

```python
import pandas as pd
import datatable as dt
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, pyqtSignal


class TableModel(QtCore.QAbstractTableModel):

    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])

            if orientation == Qt.Vertical:
                return str(self._data.index[section])
```

#### dataTable库

```python
import datatable as dt

xxxData = {"x": 1, "y": 2, "z": 4}
dt_df = dt.Frame([xxxData])
# pd_df = dt_df.to_pandas()


r = dt.Frame([xxxData for _ in range(5)])
dt_df.rbind(r)

print(dt_df[-10:, :].to_pandas())

# print(pd_df)

```

store a object

```python
class B(object):
    def __init__(self):
        self.bcc = 1
        self.bdd = 2


class A(object):
    def __init__(self):
        self.curTime: int = 1
        self.aa: List[str] = []
        self.bb: Set[str] = set()
        self.cc: List = [B()]


if __name__ == "__main__":
    a = A()
    dt_a = dt.Frame(A=[a.curTime], B=[a], types=[dt.Type.int32, dt.Type.obj64])

    a_l = dt_a.to_list()
    print(dt_a)

    print(a_l[1][0].__dict__)
```

load from json file

```python
data_json = json.loads(dataStr)
df_class = pd.json_normalize(data_json)
df = df_class.drop(columns=['xxx'])
df['curTime'] = pd.to_datetime(df['curTime'], unit='s')

dt_class = dt.Frame(df_class)

dt_class = dt.fread(data_json)

print(df_class)
print(dt_class)
```

#### data view

dataTable的显示采用QTableView，因为QTableWidget不能设置Model，model的数据可以是List、Numpy、Pandas Frame。
Numpy和Pandas类似c语言数组静态分配内存，添加行数据需要重新分配内存，然后拷贝，速度很慢。所以，采用DataTable库，DataTable Frame可以转换成pandas 和numpy。

```python
class xxxWidget(QWidget, Ui_Form):
    def init_UI(self):
        self.tabList = ["tab1", "tab2", "tab3", "tab4"]
        for tabName in self.tabList:
            self.init_Tab(tabName)

    def init_Tab(self, tabName: str):
        tab = QtWidgets.QWidget()
        tab.setObjectName(self.getTabName(tabName))
        label = QtWidgets.QLabel(tab)
        label.setGeometry(QtCore.QRect(0, 0, 761, 31))
        label.setObjectName(self.getLabelName(tabName))
        label.setText(self.getLabelText(tabName))

        tableWidget = QtWidgets.QTableView(tab)
        tableWidget.setGeometry(QtCore.QRect(0, 40, 761, 451))
        tableWidget.setObjectName(self.getTableWidgetName(tabName))
        self.tabWidget.addTab(tab, self.getTabText(tabName))
        # self.tabWidget .setTabText(self.tabWidget .indexOf(tab), self.getTabText(tabName))

    def flushTable(self):
        # find table from tab of tabName
        curTab = self.tabWidget.currentWidget()
        if curTab is None:
            return

        index = self.tabWidget.indexOf(curTab)
        tabText = self.tabWidget.tabText(index)
        curTable = curTab.findChild(QtWidgets.QTableView)
        if curTable is not None:
            data = self.storage.getTailData(tabText)
            model = TableModel(data)
            curTable.setModel(model)
```
