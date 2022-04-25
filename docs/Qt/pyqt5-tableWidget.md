
### tableWidget中插入PushButton

当在tableWidget中插入PushButton时，如果table中的数据是动态变化的直接传入row，col不会得到想要的结果，这种方式只能在回调函数中获取row，col最后一次的取值。

所以需要采用如下方式：
```
index = QtCore.QPersistentModelIndex(self.tableWidget.model().index(row_index, col_index))
remove_knowledge_btn.clicked.connect(
    lambda *args, index=index: self.remove_knowledge_button_action(index.row(), index.column(),
                                                                    display_knowledge_data))
```
例如：

```python
remove_knowledge_btn = QPushButton(item_data)
remove_knowledge_btn.setDown(True)
remove_knowledge_btn.setStyleSheet('QPushButton{margin:3px}')
resourceId = row_data[3]
remove_knowledge_btn.setAccessibleName(resourceId)

index = QtCore.QPersistentModelIndex(self.tableWidget.model().index(row_index, col_index))
remove_knowledge_btn.clicked.connect(
    lambda *args, index=index: self.remove_knowledge_button_action(index.row(), index.column(),
                                                                    display_knowledge_data))
self.tableWidget.setCellWidget(row_index, col_index, remove_knowledge_btn)
```

这样在button的回调函数中可以通过row和col值区分是表格中哪个单元格中的push_button的被clicked