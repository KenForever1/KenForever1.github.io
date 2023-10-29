
[qt6 qml book](https://www.bookstack.cn/read/qt6-qml-book/intro.md)

https://doc.qt.io/qt-6/examples-widgets.html

https://towardsdatascience.com/python-interactive-network-visualization-using-networkx-plotly-and-dash-e44749161ed7


pyqt5 conbine plotly by webengine. plotly is dynamic and interactive , matplot is static.
```python
from PyQt5 import QtCore, QtWidgets, QtWebEngineWidgets
import plotly.express as px


class Widget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.button = QtWidgets.QPushButton('Plot', self)
        self.browser = QtWebEngineWidgets.QWebEngineView(self)

        vlayout = QtWidgets.QVBoxLayout(self)
        vlayout.addWidget(self.button, alignment=QtCore.Qt.AlignHCenter)
        vlayout.addWidget(self.browser)

        self.button.clicked.connect(self.show_graph)
        self.resize(1000,800)

    def show_graph(self):
        df = px.data.tips()
        fig = px.box(df, x="day", y="total_bill", color="smoker")
        fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
        self.browser.setHtml(fig.to_html(include_plotlyjs='cdn'))

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    widget = Widget()
    widget.show()
    app.exec()
```

[qml book](https://qmlbook.github.io/index.html)

a blog https://muyuuuu.github.io/archives/


+ how to set tableView columns header resize model
```python
            model = TableModel(appData)

            curTable.setModel(model)

            # set columns resize model
            curTable.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
            curTable.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Interactive)
            curTable.setColumnWidth(0, self.parent_widget.width() // 2)
            curTable.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
            curTable.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
            curTable.horizontalHeader().setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
```

注意，必须要在model被set以后设置ResizeModel才有效果

TableModel
```python
from PyQt5.QtCore import Qt
from PyQt5 import QtCore


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