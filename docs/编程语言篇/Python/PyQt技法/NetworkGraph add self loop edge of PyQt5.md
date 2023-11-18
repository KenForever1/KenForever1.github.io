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
        self.self_arrow_size = 15

        self._source.add_edge(self)
        self._dest.add_edge(self)

        self._line = QLineF()
        self.setZValue(-1)
        self.adjust()

        self.self_loop_top = 0
        self.self_loop_left = 0
        self.self_loop_width = 30
        self.self_loop_height = 40

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

    def paint_self_loop_edge(self, painter):
        painter.setRenderHints(QPainter.Antialiasing)
        # Set the color and pen width of the ellipse
        startPos = self._source.pos() + QPointF(self._source._radius - 8, self._source._radius * 2 - 8)
        self.self_loop_left, self.self_loop_top = startPos.x(), startPos.y()
        # self._tickness is pen width
        painter.setPen(QPen(Qt.black, self._tickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
         # only draw line of ellipse, not need to set brush. it will fill ellipse, when set brush
        # set ellipse fill color to white
        # painter.setBrush(QColor(255, 255, 255, 0))
        painter.drawEllipse(self.self_loop_left, self.self_loop_top, self.self_loop_width, self.self_loop_height)

        # draw arrow in ellipse bottom point
        arrow_point = QPointF(self.self_loop_left + self.self_loop_width // 2,
                              self.self_loop_top + self.self_loop_height)

        # this can be adjusted by your self
        angle = 6.9
        arrow_p1 = arrow_point + QPointF(
            math.sin(angle + math.pi / 3) * self.self_arrow_size,
            math.cos(angle + math.pi / 3) * self.self_arrow_size,
        )
        arrow_p2 = arrow_point + QPointF(
            math.sin(angle + math.pi - math.pi / 3) * self.self_arrow_size,
            math.cos(angle + math.pi - math.pi / 3) * self.self_arrow_size,
        )
        arrow_head = QPolygonF()
        arrow_head.clear()
        arrow_head.append(arrow_point)
        arrow_head.append(arrow_p1)
        arrow_head.append(arrow_p2)
        painter.setPen(QPen(QColor(self._color), self._tickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(QBrush(QColor(self._color)))
        painter.drawPolygon(arrow_head)

    def paint(self, painter, option, widget=None):
        """Override from QGraphicsItem

        Draw Edge. This method is called from Edge.adjust()

        Args:
            painter (QPainter)
            option (QStyleOptionGraphicsItem)
        """
        # drawn self loop edge of node
        if self._source.pos() == self._dest.pos():
            self.paint_self_loop_edge(painter)
            return

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

    def __init__(self, graph: nx.DiGraph = None, parent=None):
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

        if graph is not None:
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

        # self.graphUpdateThread = GraphUpdateThread(self.graph_changed)
        # self.graphUpdateThread.start()


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
        gh.add_edges_from([("Node2", "Node3", {"color": "yellow"}), ("Node3", "Node4")])
        grhDict[index] = gh

        time.sleep(2)
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