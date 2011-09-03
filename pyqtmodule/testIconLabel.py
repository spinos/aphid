from PyQt4 import QtCore, QtGui
import iconLabel as il

if __name__ == '__main__':

    import sys

    app = QtGui.QApplication(sys.argv)

    scene = QtGui.QGraphicsScene()
    scene.setSceneRect(-200, -200, 400, 400)
    scene.setItemIndexMethod(QtGui.QGraphicsScene.NoIndex)

    desc = il.ItemDesc()
    
    desc.label_text = 'circle'
    desc.image_path = ':circle.png'
    desc.pos_x = 100
    desc.pos_y = 50
    scene.addItem(il.Square(desc))
    
    desc.label_text = 'square'
    desc.image_path = ':square.png'
    desc.pos_x = -100
    desc.pos_y = -90
    scene.addItem(il.Square(desc))
    
    
    view = QtGui.QGraphicsView(scene)
    view.setRenderHint(QtGui.QPainter.Antialiasing)
    view.setCacheMode(QtGui.QGraphicsView.CacheBackground)
    view.setViewportUpdateMode(QtGui.QGraphicsView.BoundingRectViewportUpdate)
    view.setDragMode(QtGui.QGraphicsView.ScrollHandDrag)
    view.setWindowTitle("icon label test")
    view.resize(400, 300)
    view.show()

    sys.exit(app.exec_())
