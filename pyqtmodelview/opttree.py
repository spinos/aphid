import sys
from PyQt4 import QtCore, QtGui

class OptTreeItem(object):
    def __init__(self, name, value, row, parent=None):
        self.nodeName = name
        self.nodeValue = value
        # Record the item's location within its parent.
        self.rowNumber = row
        self.parentItem = parent
        self.childItems = {}

    def node_name(self):
        return self.nodeName

    def node_value(self):
        return self.nodeValue

    def parent(self):
        return self.parentItem

    def key_child(self, i):
        row = 0
        for k in self.nodeValue.keys():
            if i == row:
                return k
            row += 1
        return None

    def child(self, i):
        if i in self.childItems:
            return self.childItems[i]

        if not isinstance(self.nodeValue, dict):
            return None

        if i < len(self.nodeValue):
            childName = self.key_child(i)
            childValue = self.nodeValue[childName]
            childItem = OptTreeItem(childName, childValue, i, self)
            self.childItems[i] = childItem
            return childItem

        return None

    def has_value(self):
        return not isinstance(self.nodeValue, dict)

    def row(self):
        return self.rowNumber


class OptTreeModel(QtCore.QAbstractItemModel):
    def __init__(self, name, settings, parent=None):
        super(OptTreeModel, self).__init__(parent)
        
        self.rootItem = OptTreeItem(name, settings, 0)

    def columnCount(self, parent):
        return 2

    def data(self, index, role):
        if not index.isValid():
            return None

        if role != QtCore.Qt.DisplayRole:
            return None

        item = index.internalPointer()

        node = item.node_value()
        attributes = []
        
        if index.column() == 0:
            return item.node_name()
        
        elif index.column() == 1:
            if item.has_value():
                attributes = item.node_value()
            else:
                attributes = ''

            return " ".join(attributes)

        return None

    def flags(self, index):
        if not index.isValid():
            return QtCore.Qt.NoItemFlags

        return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable

    def headerData(self, section, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            if section == 0:
                return "Name"

            if section == 1:
                return "Value"

        return None

    def index(self, row, column, parent):
        if not self.hasIndex(row, column, parent):
            return QtCore.QModelIndex()

        if not parent.isValid():
            parentItem = self.rootItem
        else:
            parentItem = parent.internalPointer()

        childItem = parentItem.child(row)
        if childItem:
            return self.createIndex(row, column, childItem)
        else:
            return QtCore.QModelIndex()

    def parent(self, child):
        if not child.isValid():
            return QtCore.QModelIndex()

        childItem = child.internalPointer()
        parentItem = childItem.parent()

        if not parentItem or parentItem == self.rootItem:
            return QtCore.QModelIndex()

        return self.createIndex(parentItem.row(), 0, parentItem)

    def rowCount(self, parent):
        if parent.column() > 0:
            return 0

        if not parent.isValid():
            parentItem = self.rootItem
        else:
            parentItem = parent.internalPointer()

        return len(parentItem.node_value())

    def reset_raw(self, name, raw):
        self.rootItem = OptTreeItem(name, raw, 0)

class Window(QtGui.QWidget):
    kHOMEDIR = ''
    def __init__(self):
        super(Window, self).__init__()
        model = OptTreeModel('root', {'abc':{'def':'ghi'}, 'jkl':{}}, self)
        
        view = QtGui.QTreeView()
        view.setModel(model)
        
        layout = QtGui.QVBoxLayout()
        layout.addWidget(view, 0)
        self.setLayout(layout)
        
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
