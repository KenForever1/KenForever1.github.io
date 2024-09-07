# [Convert pyQt UI to python](https://stackoverflow.com/questions/18429452/convert-pyqt-ui-to-python)
```
pyuic5 input.ui -o output.py
```

convert gpkg file to shp file
```
ogr2ogr -f "ESRI Shapefile" output_folder input.gpkg 
```


python shapely library and RTree

以下是使用Python Rtree库进行RTree空间索引的一个简单示例。

首先，需要安装Rtree库。使用以下命令在控制台中安装：

```
pip install Rtree
```

在下面的示例中，我们使用一个简单的点数据集。您需要将点坐标作为x和y值传递，我们将把它们转换为shapely库的Point实例。示例中使用的点是从csv文件中加载的。

```python
from rtree import index
from shapely.geometry import Point
import csv

# 创建空间索引
idx = index.Index()

# 从CSV文件加载点
with open('points.csv') as csvfile:
    points_reader = csv.reader(csvfile)
    for id, row in enumerate(points_reader):
        x, y = float(row[0]), float(row[1])
        point = Point(x, y)
        idx.insert(id, point.bounds)

# 查询点
query_point = Point(10, 10)
result_ids = list(idx.intersection(query_point.bounds))

# 处理结果点ID
for id in result_ids:
    print(f'Point {id} found')
```

在上面的代码中，我们首先导入了所需的库，然后创建了一个新的RTree索引idx。使用csv读取器加载点坐标，转换为Shapely Point对象，并将它们添加到Rtree索引中。

使用Shapely库中的Point，我们可以创建一个新的查询点query_point。和前面的代码类似，我们使用Shapely Point的边界属性创建一个查询范围，并使用Rtree的intersection函数找到与查询范围相交的对象。

最后，在遍历结果的点ID时，输出用于指示要素已找到的消息。

注意，上面的示例只是一个简单的演示，因此使用csv文件来加载点。在实际情况下，您可以使用QGIS中的矢量图层或使用其他技术来从数据库中加载空间对象。

空间索引算法：R树或Quadtree