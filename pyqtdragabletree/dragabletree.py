#!/usr/bin/env python

############################################################################
##
## Copyright (C) 2005-2005 Trolltech AS. All rights reserved.
##
## This file is part of the example classes of the Qt Toolkit.
##
## This file may be used under the terms of the GNU General Public
## License version 2.0 as published by the Free Software Foundation
## and appearing in the file LICENSE.GPL included in the packaging of
## this file.  Please review the following information to ensure GNU
## General Public Licensing requirements will be met:
## http://www.trolltech.com/products/qt/opensource.html
##
## If you are unsure which license is appropriate for your use, please
## review the following information:
## http://www.trolltech.com/products/qt/licensing.html or contact the
## sales department at sales@trolltech.com.
##
## This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
## WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
##
############################################################################

from PyQt4 import QtCore, QtGui

class DragableItem(QtGui.QTreeWidgetItem):
    def __init__(self, parent=None):
        super(DragableItem, self).__init__(parent)
        self.name = 'foo'
    def set_name(self, name):
        self.name = name
        self.setText(0, self.name)
        
    
class SimpleTree(QtGui.QTreeWidget):
    def __init__(self, parent=None):
        super(SimpleTree, self).__init__()
        spawn1 = DragableItem(self)
        spawn1.set_name('A1')
        spawn2 = DragableItem(self)
        spawn2.set_name('A2')
        spawn3 = DragableItem(self)
        spawn3.set_name('A3')
        
        spawn11 = DragableItem(spawn1)
        spawn11.set_name('B1')
        spawn12 = DragableItem(spawn1)
        spawn12.set_name('B2')
        spawn23 = DragableItem(spawn2)
        spawn23.set_name('B3')
        
        self.expandItem(spawn1)
        self.expandItem(spawn2)
        
        self.setAcceptDrops(True)
        
    def mouseMoveEvent(self, event):
        child = self.itemAt(event.pos())
        if not child:
            return
            
        print 'start dragging', child.text(0)
        
        itemData = QtCore.QByteArray()
        dataStream = QtCore.QDataStream(itemData, QtCore.QIODevice.WriteOnly)
        dataValue = QtCore.QStringList()
        dataValue << child.text(0)
        dataStream << dataValue
 
        mimeData = QtCore.QMimeData()
        mimeData.setData('application/x-dnditemdata', itemData)
 
        drag = QtGui.QDrag(self)
        drag.setMimeData(mimeData)
        drag.setHotSpot(event.pos())

        if drag.exec_(QtCore.Qt.CopyAction | QtCore.Qt.MoveAction, QtCore.Qt.CopyAction) == QtCore.Qt.MoveAction:
            pass
            
    def dropEvent(self, event):
        if not event.mimeData().hasFormat('application/x-dnditemdata'):
            event.ignore()
            
        itemData = event.mimeData().data('application/x-dnditemdata')
        dataStream = QtCore.QDataStream(itemData, QtCore.QIODevice.ReadOnly)
            
        sourceAttrib = QtCore.QStringList()
        dataStream >> sourceAttrib

        hit = self.itemAt(event.pos())
        
        if not hit:
            print 'hit nothing'
            event.ignore()
            
        print 'drop %s onto %s' % ( sourceAttrib[0], hit.text(0) )
        
        
        spawn = DragableItem(hit)
        spawn.set_name(sourceAttrib[0])
        self.expandItem(hit)
        
        if event.source() == self:
            event.setDropAction(QtCore.Qt.MoveAction)
            event.accept()
        else:
            event.acceptProposedAction()
            
    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat('application/x-dnditemdata'):
            if event.source() == self:
                event.setDropAction(QtCore.Qt.MoveAction)
                event.accept()
            else:
                event.acceptProposedAction()
        else:
            event.ignore()
            
    dragMoveEvent = dragEnterEvent
            
if __name__ == '__main__':

    import sys

    app = QtGui.QApplication(sys.argv)

    view = SimpleTree()
    view.setWindowTitle("Dragable Tree")
    view.show()
    sys.exit(app.exec_())
