from PyQt4 import QtCore, QtGui
from FooImage import FooImage

class RenderArea(QtGui.QWidget):
    def __init__(self, parent=None):
        super(RenderArea, self).__init__(parent)
        
        self.imageLabel = QtGui.QImage()
        self.imageWidth = 1280
        self.imageHeight = 720
        self.fKd = 1.0
        self.a = FooImage('hello')
        
        

    def open(self):
        fileName = QtGui.QFileDialog.getOpenFileName(self, "Open File",
                QtCore.QDir.currentPath(), "Image files (*.tif *.jpg *.png)")
        if fileName:
            image = QtGui.QImage(fileName)
            if image.isNull():
                QtGui.QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return
                
            if image.width() > 1280:
                image = image.scaledToWidth(1280)
            if image.height() > 720: 
            	image = image.scaledToHeight(720)

            self.imageLabel = image.convertToFormat(QtGui.QImage.Format_RGB32)
            self.imageWidth = image.width()
            self.imageHeight = image.height()
            
            self.a.setSize(image.width(), image.height())
            self.a.addColor(image.bits())


    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        painter.fillRect(event.rect(), QtGui.QBrush(QtCore.Qt.white))
        
        
        self.a.setRed(self.fKd);
        
        image = QtGui.QImage(self.a.display() ,self.imageWidth, self.imageHeight, QtGui.QImage.Format_RGB32)

	painter.drawImage(0, 0, image)
	
    def setXRotation(self, angle):
    	self.fKd = angle / 255.0
        self.update()
