
```python
import math
import sys
from typing import Dict

from PyQt5.QtCore import (QEasingCurve, QLineF,
                          QParallelAnimationGroup, QPointF,
                          QPropertyAnimation, QRectF, Qt)
from PyQt5.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import (QApplication, QComboBox, QGraphicsItem,
                             QGraphicsObject, QGraphicsScene, QGraphicsView,
                             QStyleOptionGraphicsItem, QVBoxLayout, QWidget)

import networkx as nx


class Node(QGraphicsObject):
    """A QGraphicsItem representing node in a graph"""

    def __init__(self, name: str, nodeArgs: Dict = None, parent=None):
        """Node constructor

        Args:
            name (str): Node label
        """
        super().__init__(parent)
        self._name = name
        self._edges = []
        if nodeArgs.get("color") is None:
            self._color = "#5AD469"
        else:
            self._color = nodeArgs.get("color")
        self._radius = 30
        self._rect = QRectF(0, 0, self._radius * 2, self._radius * 2)

        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)

    def boundingRect(self) -> QRectF:
        """Override from QGraphicsItem

        Returns:
            QRect: Return node bounding rect
        """
        return self._rect

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget = None):
        """Override from QGraphicsItem

        Draw node

        Args:
            painter (QPainter)
            option (QStyleOptionGraphicsItem)
        """
        painter.setRenderHints(QPainter.Antialiasing)
        painter.setPen(
            QPen(
                QColor(self._color).darker(),
                2,
                Qt.SolidLine,
                Qt.RoundCap,
                Qt.RoundJoin,
            )
        )
        painter.setBrush(QBrush(QColor(self._color)))
        painter.drawEllipse(self.boundingRect())
        painter.setPen(QPen(QColor("white")))
        painter.drawText(self.boundingRect(), Qt.AlignCenter, self._name)

    def add_edge(self, edge):
        """Add an edge to this node

        Args:
            edge (Edge)
        """
        self._edges.append(edge)

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value):
        """Override from QGraphicsItem

        Args:
            change (QGraphicsItem.GraphicsItemChange)
            value (Any)

        Returns:
            Any
        """
        if change == QGraphicsItem.ItemPositionHasChanged:
            for edge in self._edges:
                edge.adjust()

        return super().itemChange(change, value)


class Edge(QGraphicsItem):
    def __init__(self, source, dest, edgeArgs, parent=None):
        """Edge constructor

        Args:
            source (Node): source node
            dest (Node): destination node
        """
        super().__init__(parent)
        self._source = source
        self._dest = dest

        self._tickness = 2

        if edgeArgs.get("color") is None:
            self._color = QColor("#2BB53C")
        else:
            self._color = edgeArgs.get("color")

        self._arrow_size = 20

        self._source.add_edge(self)
        self._dest.add_edge(self)

        self._line = QLineF()
        self.setZValue(-1)
        self.adjust()

    def boundingRect(self):
        """Override from QGraphicsItem

        Returns:
            QRectF: Return node bounding rect
        """
        return QRectF(self._line.p1(), self._line.p2()).normalized().adjusted(
            -self._tickness - self._arrow_size,
            -self._tickness - self._arrow_size,
            self._tickness + self._arrow_size,
            self._tickness + self._arrow_size,
        )

    def adjust(self):
        """
        Update edge position from source and destination node.
        This method is called from Node::itemChange
        """
        self.prepareGeometryChange()
        self._line = QLineF(
            self._source.pos() + self._source.boundingRect().center(),
            self._dest.pos() + self._dest.boundingRect().center(),
        )

    def _draw_arrow(self, painter, start, end):
        """Draw arrow from start point to end point.

        Args:
            painter (QPainter)
            start (QPointF): start position
            end (QPointF): end position
        """
        painter.setBrush(QBrush(QColor(self._color)))

        line = QLineF(end, start)

        angle = math.atan2(-line.dy(), line.dx())
        arrow_p1 = line.p1() + QPointF(
            math.sin(angle + math.pi / 3) * self._arrow_size,
            math.cos(angle + math.pi / 3) * self._arrow_size,
        )
        arrow_p2 = line.p1() + QPointF(
            math.sin(angle + math.pi - math.pi / 3) * self._arrow_size,
            math.cos(angle + math.pi - math.pi / 3) * self._arrow_size,
        )

        arrow_head = QPolygonF()
        arrow_head.clear()
        arrow_head.append(line.p1())
        arrow_head.append(arrow_p1)
        arrow_head.append(arrow_p2)
        painter.drawLine(line)
        painter.drawPolygon(arrow_head)

    def _arrow_target(self):
        """Calculate the position of the arrow taking into account the size of the destination node

        Returns:
            QPointF
        """
        target = self._line.p1()
        center = self._line.p2()
        radius = self._dest._radius
        vector = target - center
        length = math.sqrt(vector.x() ** 2 + vector.y() ** 2)
        if length == 0:
            return target
        normal = vector / length
        target = QPointF(center.x() + (normal.x() * radius), center.y() + (normal.y() * radius))

        return target

    def paint(self, painter, option, widget=None):
        """Override from QGraphicsItem

        Draw Edge. This method is called from Edge.adjust()

        Args:
            painter (QPainter)
            option (QStyleOptionGraphicsItem)
        """

        if self._source and self._dest:
            painter.setRenderHints(QPainter.Antialiasing)

            painter.setPen(
                QPen(
                    QColor(self._color),
                    self._tickness,
                    Qt.SolidLine,
                    Qt.RoundCap,
                    Qt.RoundJoin,
                )
            )
            painter.drawLine(self._line)
            self._draw_arrow(painter, self._line.p1(), self._arrow_target())
            self._arrow_target()


class GraphView(QGraphicsView):
    def __init__(self, graph: nx.DiGraph, parent=None):
        """GraphView constructor

        This widget can display a directed graph

        Args:
            graph (nx.DiGraph): a networkx directed graph
        """
        super().__init__()
        self._graph = graph
        self._scene = QGraphicsScene()
        self.setScene(self._scene)

        # Used to add space between nodes
        self._graph_scale = 200

        # Map node name to Node object {str=>Node}
        self._nodes_map = {}

        # List of networkx layout function
        self._nx_layout = {
            "circular": nx.circular_layout,
            "planar": nx.planar_layout,
            "random": nx.random_layout,
            "shell_layout": nx.shell_layout,
            "kamada_kawai_layout": nx.kamada_kawai_layout,
            "spring_layout": nx.spring_layout,
            "spiral_layout": nx.spiral_layout,
        }

        self._load_graph()
        self.set_nx_layout("circular")

    def get_nx_layouts(self) -> list:
        """Return all layout names

        Returns:
            list: layout name (str)
        """
        return self._nx_layout.keys()

    def set_nx_layout(self, name: str):
        """Set networkx layout and start animation

        Args:
            name (str): Layout name
        """
        if name in self._nx_layout:
            self._nx_layout_function = self._nx_layout[name]

            # Compute node position from layout function
            positions = self._nx_layout_function(self._graph)

            # Change position of all nodes using an animation
            self.animations = QParallelAnimationGroup()
            for node, pos in positions.items():
                x, y = pos
                x *= self._graph_scale
                y *= self._graph_scale
                item = self._nodes_map[node]

                animation = QPropertyAnimation(item, b"pos")
                animation.setDuration(1000)
                animation.setEndValue(QPointF(x, y))
                animation.setEasingCurve(QEasingCurve.OutExpo)
                self.animations.addAnimation(animation)

            self.animations.start()

    def _load_graph(self):
        """Load graph into QGraphicsScene using Node class and Edge class"""

        self.scene().clear()
        self._nodes_map.clear()

        # Add nodes
        for node in self._graph:
            nodeAttr = self._graph.nodes.get(node)
            item = Node(node, nodeAttr)
            self.scene().addItem(item)
            self._nodes_map[node] = item

        # Add edges
        for a, b, attr in self._graph.edges(data=True):
            source = self._nodes_map[a]
            dest = self._nodes_map[b]
            edgeArgs = attr
            self.scene().addItem(Edge(source, dest, edgeArgs))


class MainWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__()

        self.graph = nx.DiGraph()

        self.graph.add_node("1", color="red")

        self.graph.add_edges_from(
            [
                ("1", "2", {"color": "blue"}),
                ("2", "1"),
                ("2", "3"),
                ("3", "4"),
                ("1", "5"),
                ("1", "6"),
                ("1", "7"),
            ]
        )

        self.view = GraphView(self.graph)
        self.choice_combo = QComboBox()
        self.choice_combo.addItems(self.view.get_nx_layouts())
        v_layout = QVBoxLayout(self)
        v_layout.addWidget(self.choice_combo)
        v_layout.addWidget(self.view)
        self.choice_combo.currentTextChanged.connect(self.view.set_nx_layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Create a networkx graph

    widget = MainWindow()
    widget.show()
    widget.resize(800, 600)
    sys.exit(app.exec())

```

[PySide6 Example](https://doc.qt.io/qtforpython/examples/example_external__networkx.html)

+ add update graph function, and use signal emit graph update
```python
import math
import sys
import time
from typing import Dict

from PyQt5.QtCore import (QEasingCurve, QLineF,
                          QParallelAnimationGroup, QPointF,
                          QPropertyAnimation, QRectF, Qt, pyqtSignal, QThread)
from PyQt5.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import (QApplication, QComboBox, QGraphicsItem,
                             QGraphicsObject, QGraphicsScene, QGraphicsView,
                             QStyleOptionGraphicsItem, QVBoxLayout, QWidget)

import networkx as nx


class Node(QGraphicsObject):
    """A QGraphicsItem representing node in a graph"""

    def __init__(self, name: str, nodeArgs: Dict = None, parent=None):
        """Node constructor

        Args:
            name (str): Node label
        """
        super().__init__(parent)
        self._name = name
        self._edges = []
        if nodeArgs.get("color") is None:
            self._color = "#5AD469"
        else:
            self._color = nodeArgs.get("color")
        self._radius = 30
        self._rect = QRectF(0, 0, self._radius * 2, self._radius * 2)

        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)

    def boundingRect(self) -> QRectF:
        """Override from QGraphicsItem

        Returns:
            QRect: Return node bounding rect
        """
        return self._rect

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget = None):
        """Override from QGraphicsItem

        Draw node

        Args:
            painter (QPainter)
            option (QStyleOptionGraphicsItem)
        """
        painter.setRenderHints(QPainter.Antialiasing)
        painter.setPen(
            QPen(
                QColor(self._color).darker(),
                2,
                Qt.SolidLine,
                Qt.RoundCap,
                Qt.RoundJoin,
            )
        )
        painter.setBrush(QBrush(QColor(self._color)))
        painter.drawEllipse(self.boundingRect())
        painter.setPen(QPen(QColor("white")))
        painter.drawText(self.boundingRect(), Qt.AlignCenter, self._name)

    def add_edge(self, edge):
        """Add an edge to this node

        Args:
            edge (Edge)
        """
        self._edges.append(edge)

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value):
        """Override from QGraphicsItem

        Args:
            change (QGraphicsItem.GraphicsItemChange)
            value (Any)

        Returns:
            Any
        """
        if change == QGraphicsItem.ItemPositionHasChanged:
            for edge in self._edges:
                edge.adjust()

        return super().itemChange(change, value)


class Edge(QGraphicsItem):
    def __init__(self, source, dest, edgeArgs, parent=None):
        """Edge constructor

        Args:
            source (Node): source node
            dest (Node): destination node
        """
        super().__init__(parent)
        self._source = source
        self._dest = dest

        self._tickness = 2

        if edgeArgs.get("color") is None:
            self._color = QColor("#2BB53C")
        else:
            self._color = edgeArgs.get("color")

        self._arrow_size = 20

        self._source.add_edge(self)
        self._dest.add_edge(self)

        self._line = QLineF()
        self.setZValue(-1)
        self.adjust()

    def boundingRect(self):
        """Override from QGraphicsItem

        Returns:
            QRectF: Return node bounding rect
        """
        return QRectF(self._line.p1(), self._line.p2()).normalized().adjusted(
            -self._tickness - self._arrow_size,
            -self._tickness - self._arrow_size,
            self._tickness + self._arrow_size,
            self._tickness + self._arrow_size,
        )

    def adjust(self):
        """
        Update edge position from source and destination node.
        This method is called from Node::itemChange
        """
        self.prepareGeometryChange()
        self._line = QLineF(
            self._source.pos() + self._source.boundingRect().center(),
            self._dest.pos() + self._dest.boundingRect().center(),
        )

    def _draw_arrow(self, painter, start, end):
        """Draw arrow from start point to end point.

        Args:
            painter (QPainter)
            start (QPointF): start position
            end (QPointF): end position
        """
        painter.setBrush(QBrush(QColor(self._color)))

        line = QLineF(end, start)

        angle = math.atan2(-line.dy(), line.dx())
        arrow_p1 = line.p1() + QPointF(
            math.sin(angle + math.pi / 3) * self._arrow_size,
            math.cos(angle + math.pi / 3) * self._arrow_size,
        )
        arrow_p2 = line.p1() + QPointF(
            math.sin(angle + math.pi - math.pi / 3) * self._arrow_size,
            math.cos(angle + math.pi - math.pi / 3) * self._arrow_size,
        )

        arrow_head = QPolygonF()
        arrow_head.clear()
        arrow_head.append(line.p1())
        arrow_head.append(arrow_p1)
        arrow_head.append(arrow_p2)
        painter.drawLine(line)
        painter.drawPolygon(arrow_head)

    def _arrow_target(self):
        """Calculate the position of the arrow taking into account the size of the destination node

        Returns:
            QPointF
        """
        target = self._line.p1()
        center = self._line.p2()
        radius = self._dest._radius
        vector = target - center
        length = math.sqrt(vector.x() ** 2 + vector.y() ** 2)
        if length == 0:
            return target
        normal = vector / length
        target = QPointF(center.x() + (normal.x() * radius), center.y() + (normal.y() * radius))

        return target

    def paint(self, painter, option, widget=None):
        """Override from QGraphicsItem

        Draw Edge. This method is called from Edge.adjust()

        Args:
            painter (QPainter)
            option (QStyleOptionGraphicsItem)
        """

        if self._source and self._dest:
            painter.setRenderHints(QPainter.Antialiasing)

            painter.setPen(
                QPen(
                    QColor(self._color),
                    self._tickness,
                    Qt.SolidLine,
                    Qt.RoundCap,
                    Qt.RoundJoin,
                )
            )
            painter.drawLine(self._line)
            self._draw_arrow(painter, self._line.p1(), self._arrow_target())
            self._arrow_target()


class GraphView(QGraphicsView):

    def __init__(self, graph: nx.DiGraph, parent=None):
        """GraphView constructor

        This widget can display a directed graph

        Args:
            graph (nx.DiGraph): a networkx directed graph
        """
        super().__init__()
        self._graph = graph
        self._scene = QGraphicsScene()
        self.setScene(self._scene)

        # Used to add space between nodes
        self._graph_scale = 200

        # Map node name to Node object {str=>Node}
        self._nodes_map = {}
        self._load_graph()
        self.set_nx_layout_circulator()

    def set_nx_layout_circulator(self):
        """Set networkx layout and start animation
        """
        self._nx_layout_function = nx.circular_layout

        # Compute node position from layout function
        positions = self._nx_layout_function(self._graph)

        # Change position of all nodes using an animation
        self.animations = QParallelAnimationGroup()
        for node, pos in positions.items():
            x, y = pos
            x *= self._graph_scale
            y *= self._graph_scale
            item = self._nodes_map[node]

            animation = QPropertyAnimation(item, b"pos")
            animation.setDuration(1000)
            animation.setEndValue(QPointF(x, y))
            animation.setEasingCurve(QEasingCurve.OutExpo)
            self.animations.addAnimation(animation)

        self.animations.start()

    def _load_graph(self):
        """Load graph into QGraphicsScene using Node class and Edge class"""
        self.scene().clear()
        self._nodes_map.clear()

        # Add nodes
        for node in self._graph:
            nodeAttr = self._graph.nodes.get(node)
            item = Node(node, nodeAttr)
            self.scene().addItem(item)
            self._nodes_map[node] = item

        # Add edges
        for a, b, attr in self._graph.edges(data=True):
            source = self._nodes_map[a]
            dest = self._nodes_map[b]
            edgeArgs = attr
            self.scene().addItem(Edge(source, dest, edgeArgs))

    def update_graph(self, graph):
        """Clear scene, load new graph, and set layout"""
        self._graph = graph
        self._load_graph()
        self.set_nx_layout_circulator()


class MainWindow(QWidget):
    graph_changed = pyqtSignal(nx.DiGraph)

    def __init__(self, parent=None):
        super().__init__()
        self.graph = nx.DiGraph()
        self.graph.add_node("1", color="red")
        self.graph.add_edges_from(
            [
                ("1", "2", {"color": "blue"}),
                ("2", "1"),
                ("2", "3"),
                ("3", "4"),
                ("1", "5"),
                ("1", "6"),
                ("1", "7"),
            ]
        )
        self.view = GraphView(self.graph)
        v_layout = QVBoxLayout(self)
        v_layout.addWidget(self.view)

        self.graph_changed.connect(self.view.update_graph)

        self.graphUpdateThread = GraphUpdateThread(self.graph_changed)
        self.graphUpdateThread.start()


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
        gh = nx.DiGraph()
        gh.add_edges_from([("1", "2"), ("2", "3")])
        grhDict[index] = gh
        index += 1

        gh = nx.DiGraph()
        gh.add_edges_from([("1", "2"), ("2", "3"), ("3", "4"), ("4", "1")])
        grhDict[index] = gh

        index += 1
        gh = nx.DiGraph()
        gh.add_edges_from([("1", "2"), ("2", "3"), ("3", "4"), ("2", "1")])
        grhDict[index] = gh

        for i in range(10):
            index = i % 3 + 1
            ph = grhDict[index]
            self.graph_changed.emit(ph)
            time.sleep(1)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Create a networkx graph

    widget = MainWindow()
    widget.show()
    widget.resize(800, 600)
    sys.exit(app.exec())

```

[nx plot](https://networkx.org/documentation/latest/auto_examples/drawing/plot_ego_graph.html)

an example draw networkx graph with Qt5

```python 
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QVBoxLayout

# class NodeGraph(object):
#     def __int__(self):


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Create a QWidget object and add a QVBoxLayout to it
        centralWidget = QWidget(self)
        layout = QVBoxLayout(centralWidget)

        # Create a FigureCanvas object and add it to the QVBoxLayout
        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        # Create an nx graph and draw it using matplotlib
        G = nx.DiGraph()
        G.add_node("1", color="red")
        G.add_edges_from(
            [
                ("1", "2", {"color": "blue"}),
                ("2", "1"),
                ("2", "2"),
                ("2", "3"),
                ("3", "4"),
                ("1", "5"),
                ("1", "6"),
                ("1", "7"),
            ]
        )
        pos = nx.circular_layout(G)
        # nx.draw_networkx_nodes(G, pos, ax=ax)
        # nx.draw_networkx_labels(G, pos, ax=ax)
        # nx.draw_networkx_edges(G, pos, ax=ax)
        # nx.draw_networkx(G, pos, ax=ax)
        nx.draw_networkx(G, pos, ax=ax,node_size = 800, node_color = "white", edge_color = "red", width=1.0)
        ax.set_axis_off()

        # Set the FigureCanvas object as the central widget of the QMainWindow
        self.setCentralWidget(centralWidget)

        # reload the Graph
        # ax.cla()
        # G = nx.DiGraph()
        # G.add_edges_from([("1", "2"), ("2", "3"), ("3", "4"), ("4", "1")])
        # pos = nx.circular_layout(G)
        # # nx.draw_networkx_nodes(G, pos, ax=ax)
        # # nx.draw_networkx_labels(G, pos, ax=ax)
        # # nx.draw_networkx_edges(G, pos, ax=ax)
        # nx.draw_networkx(G, pos, ax=ax)
        # ax.set_axis_off()
        # canvas.draw()

if __name__ == '__main__':
    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec_()

```


```python
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QVBoxLayout


class NetGraph2(object):
    def __init__(self, parent_layout, G: nx.DiGraph = None):
        self.layout = parent_layout
        # Create a FigureCanvas object and add it to the QVBoxLayout
        self.fig, self.ax = plt.subplots()
        self.fig.set_facecolor('#001528')
        self.fig.set_figwidth(5)
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)
        self.edge_color = "#4DCAD2"
        self.node_color = "#70CFED"
        self.line_width = 1.5
        self.node_size = 600

        if G is not None:
            # Create an nx graph and draw it using matplotlib
            # G = nx.DiGraph()
            # G.add_node("1", color="red")
            # G.add_edges_from(
            #     [
            #         ("1", "2", {"color": "blue"}),
            #         ("2", "1"),
            #         ("2", "2"),
            #         ("2", "3"),
            #         ("3", "4"),
            #         ("1", "5"),
            #         ("1", "6"),
            #         ("1", "7"),
            #     ]
            # )
            pos = nx.circular_layout(G)
            # nx.draw_networkx_nodes(G, pos, ax=ax)
            # nx.draw_networkx_labels(G, pos, ax=ax)
            # nx.draw_networkx_edges(G, pos, ax=ax)
            # nx.draw_networkx(G, pos, ax=ax)
            nx.draw_networkx(G, pos, ax=self.ax,
                             node_size=self.node_size,
                             node_color=self.node_color,
                             edge_color=self.edge_color,
                             width=self.line_width)
            self.ax.set_axis_off()

    def update_graph(self, G: nx.DiGraph):
        # reload the Graph
        self.ax.cla()
        pos = nx.circular_layout(G)
        # nx.draw_networkx_nodes(G, pos, ax=ax)
        # nx.draw_networkx_labels(G, pos, ax=ax)
        # nx.draw_networkx_edges(G, pos, ax=ax)
        nx.draw_networkx(G, pos, ax=self.ax,
                         node_size=self.node_size,
                         node_color=self.node_color,
                         edge_color=self.edge_color,
                         width=self.line_width,
                         arrowsize=15,
                         font_size=16
                         # style='dashed'
                         )
        self.ax.set_axis_off()
        self.canvas.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Create a QWidget object and add a QVBoxLayout to it
        centralWidget = QWidget(self)
        layout = QVBoxLayout(centralWidget)
        G = nx.DiGraph()
        G.add_node("1", color="red")
        G.add_edges_from(
            [
                ("1", "2", {"color": "blue"}),
                ("2", "1"),
                ("2", "2"),
                ("2", "3"),
                ("3", "4"),
                ("1", "5"),
                ("1", "6"),
                ("1", "7"),
            ]
        )
        view = NetGraph2(layout, G)

        # Set the FigureCanvas object as the central widget of the QMainWindow
        self.setCentralWidget(centralWidget)


if __name__ == '__main__':
    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec_()

```