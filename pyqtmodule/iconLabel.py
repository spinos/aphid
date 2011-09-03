from PyQt4 import QtCore, QtGui

import iconLabel_rc

class ItemDesc(object):
    dim_x = 64
    dim_y = 64
    pos_x = 0
    pos_y = 0
    angle = 0.0
    image_path = ''
    label_text = ''
    
class Square(QtGui.QGraphicsItem):
    adjust = 1.0
    def __init__(self, desc):
        super(Square, self).__init__()
        self.color = QtCore.Qt.gray
        self.title = desc.label_text
        self.width = desc.dim_x
        self.height = desc.dim_y
        self.image = QtGui.QPixmap(desc.image_path)
        self.bbox = QtCore.QRectF(-(self.width+self.adjust)/2, -(self.height+self.adjust)/2, self.width + self.adjust, self.height + self.adjust)
        self.setPos(desc.pos_x, desc.pos_y)
             
    def boundingRect(self):
        return self.bbox
        
    def shape(self):
        path = QtGui.QPainterPath()
        path.addRect(self.boundingRect())
        return path;
        
    def paint(self, painter, option, widget):
        painter.setBrush(self.color)
        target = QtCore.QRectF(-32.0, -32.0, 64.0, 64.0);
        source = QtCore.QRectF(-0.0, -0.0, 64.0, 64.0);
        painter.drawPixmap(target, self.image, source);
        painter.drawText(QtCore.QPoint(-self.width/2 + 5, -self.height/2 + 15), self.title)
        
    
        
if __name__ == '__main__':

    import sys

    app = QtGui.QApplication(sys.argv)
    
    scene = QtGui.QGraphicsScene()
    scene.setSceneRect(-200, -200, 400, 400)
    scene.setItemIndexMethod(QtGui.QGraphicsScene.NoIndex)

    desc = ItemDesc()
    
    desc.label_text = 'circle'
    desc.image_path = ':circle.png'
    desc.pos_x = 100
    desc.pos_y = 50
    scene.addItem(Square(desc))
    
    desc.label_text = 'square'
    desc.image_path = ':square.png'
    desc.pos_x = -100
    desc.pos_y = -90
    scene.addItem(Square(desc))
    
    
    view = QtGui.QGraphicsView(scene)
    view.setRenderHint(QtGui.QPainter.Antialiasing)
    view.setCacheMode(QtGui.QGraphicsView.CacheBackground)
    view.setViewportUpdateMode(QtGui.QGraphicsView.BoundingRectViewportUpdate)
    view.setDragMode(QtGui.QGraphicsView.ScrollHandDrag)
    view.setWindowTitle("icon label test")
    view.resize(400, 300)
    view.show()

    sys.exit(app.exec_())
