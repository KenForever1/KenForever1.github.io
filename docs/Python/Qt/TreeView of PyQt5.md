
```python
import time

from PyQt5.QtCore import Qt, QModelIndex, QThread, pyqtSignal
from PyQt5.QtGui import QStandardItem, QStandardItemModel, QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QTreeView, QHeaderView
import sys

from typing import List


class NodeInfo(object):
    def __init__(self, nodeName=None, nodeState=None):
        self.nodeName: str = nodeName
        self.nodeState: str = nodeState


class TaskGroupInfo(object):
    def __init__(self, taskName=None, taskId=None, leadIdx=None, nodes=None):
        self.taskName: str = taskName
        self.taskId: int = taskId
        self.leadIdx: int = leadIdx
        self.nodes: List[NodeInfo] = nodes


class TreeData(object):
    def __init__(self, taskGroupInfoList):
        self.taskGroupInfoList: List[TaskGroupInfo] = taskGroupInfoList


class GraphUpdateThread(QThread):
    def __init__(self, signal):
        super(GraphUpdateThread, self).__init__()
        self.graph_changed = signal

    def __del__(self):
        # self.wait()
        pass

    def run(self):
        grhDict = {}
        index = 1
        tree_data = TreeData([
            TaskGroupInfo("Group 1", 1, 1, [
                NodeInfo("Node 1", "On"),
                NodeInfo("Node 2", "Off"),
                NodeInfo("Node 3", "On"),
                NodeInfo("Node 3", "On"),
                NodeInfo("Node 3", "On")
            ]),
            TaskGroupInfo("Group 2", 2, 2, [
                NodeInfo("Node 4", "Off"),
                NodeInfo("Node 5", "On"),
                NodeInfo("Node 6", "Off")
            ]),
            TaskGroupInfo("Group 2", 2, 2, [
                NodeInfo("Node 4", "Off"),
                NodeInfo("Node 5", "On"),
                NodeInfo("Node 6", "Off")
            ]),
            TaskGroupInfo("Group 2", 2, 2, [
                NodeInfo("Node 4", "Off"),
                NodeInfo("Node 5", "On"),
                NodeInfo("Node 6", "Off")
            ])
        ])
        grhDict[index] = tree_data
        index += 1

        tree_data = TreeData([
            TaskGroupInfo("Group 1", 1, 1, [
                NodeInfo("Node 1", "On"),
                NodeInfo("Node 2", "Off"),
                NodeInfo("Node 3", "On"),
                NodeInfo("Node 3", "On"),
                NodeInfo("Node 3", "On")
            ])
        ])
        grhDict[index] = tree_data
        index += 1

        tree_data = TreeData([
            TaskGroupInfo("Group 1", 1, 1, [
                NodeInfo("Node 1", "On"),
                NodeInfo("Node 2", "Off"),
                NodeInfo("Node 3", "On"),
                NodeInfo("Node 3", "On"),
                NodeInfo("Node 3", "On")
            ]),
            TaskGroupInfo("Group 2", 2, 2, [
                NodeInfo("Node 4", "Off"),
                NodeInfo("Node 5", "On"),
                NodeInfo("Node 6", "Off")
            ]),
            TaskGroupInfo("Group 2", 2, 2, [
                NodeInfo("Node 4", "Off"),
                NodeInfo("Node 5", "On"),
                NodeInfo("Node 6", "Off")
            ]),
            TaskGroupInfo("Group 2", 2, 2, [
                NodeInfo("Node 4", "Off"),
                NodeInfo("Node 5", "On"),
                NodeInfo("Node 6", "Off")
            ])
            , TaskGroupInfo("Group 2", 2, 2, [
                NodeInfo("Node 4", "Off"),
                NodeInfo("Node 5", "On"),
                NodeInfo("Node 6", "Off")
            ]),
            TaskGroupInfo("Group 2", 2, 2, [
                NodeInfo("Node 4", "Off"),
                NodeInfo("Node 5", "On"),
                NodeInfo("Node 6", "Off")
            ]),
            TaskGroupInfo("Group 2", 2, 2, [
                NodeInfo("Node 4", "Off"),
                NodeInfo("Node 5", "On"),
                NodeInfo("Node 6", "Off")
            ])
        ])
        grhDict[index] = tree_data
        index += 1

        time.sleep(2)
        for i in range(10):
            index = i % 3 + 1
            ph = grhDict[index]
            self.graph_changed.emit(ph)
            time.sleep(1)


class TaskGroupInfoTree(QTreeView):
    def __init__(self, tree_data):
        super().__init__()

        # Create a QTreeView widget and set its model
        self.setGeometry(100, 100, 800, 600)
        # self.tree_view.setIconSize(Qt.SizeHint(16, 16))
        self.setRootIsDecorated(False)
        self.setAlternatingRowColors(True)
        self.setAnimated(True)
        self.setHeaderHidden(True)
        self.setHeaderHidden(True)

        self.header = QHeaderView(Qt.Horizontal, self)
        self.setHeader(self.header)

        self.model = QStandardItemModel()
        self.setModel(self.model)

        # Create a root item for the model
        self.root_item = self.model.invisibleRootItem()

        # Populate the model with the tree data
        self.populate_model(tree_data, self.root_item)

        self.expandAll()

    def populate_model(self, tree_data, parent_item):
        for task_group_info in tree_data.taskGroupInfoList:
            task_group_item = QStandardItem(QIcon("group.png"), f"{task_group_info.taskName}")
            task_group_item.setEditable(False)
            parent_item.appendRow(task_group_item)
            for node_info in task_group_info.nodes:
                node_item = QStandardItem(QIcon("node.png"), f"{node_info.nodeName}")
                node_item.setEditable(False)
                task_group_item.appendRow(node_item)

    def update_group_info(self, new_tree_data):
        # Clear the current model
        self.model.clear()

        self.root_item = self.model.invisibleRootItem()

        # Repopulate the model with the new tree data
        self.populate_model(new_tree_data, self.root_item)

        # Expand all items in the tree view
        self.expandAll()


class MainWindow(QMainWindow):
    tree_changed = pyqtSignal(TreeData)

    def __init__(self):
        super().__init__()

        # Set the window title and geometry
        self.setWindowTitle("PyQt5 Tree View")
        self.setGeometry(100, 100, 800, 600)


        tree_data = TreeData([
            TaskGroupInfo("Group 1", 1, 1, [
                NodeInfo("Node 1", "On"),
                NodeInfo("Node 2", "Off"),
                NodeInfo("Node 3", "On")
            ]),
            TaskGroupInfo("Group 2", 2, 2, [
                NodeInfo("Node 4", "Off"),
                NodeInfo("Node 5", "On"),
                NodeInfo("Node 6", "Off")
            ])
        ])

        self.tree_view = TaskGroupInfoTree(tree_data)


        # Show the tree view widget
        self.setCentralWidget(self.tree_view)

        self.tree_changed.connect(self.tree_view.update_group_info)

        self.dataGenThread = GraphUpdateThread(self.tree_changed)
        self.dataGenThread.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

```