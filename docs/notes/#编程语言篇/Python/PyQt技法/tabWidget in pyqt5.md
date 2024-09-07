+ 如何根据click index 获取tab
```
    @pyqtSlot(int)
    def on_tabWidgetOfCmd_tabBarClicked(self, index):
        """
        Slot documentation goes here.
        
        @param index DESCRIPTION
        @type int
        """
        self.curAppName = self.getCurAppNameByTabIndex(index)
        print("on_tabWidgetOfCmd_currentChanged curAppName : " + self.curAppName)

        tab = self.tabWidgetOfCmd.widget(index)

```
+ 