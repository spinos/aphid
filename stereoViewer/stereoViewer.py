#
# Simple Image Viewer using QImage
#

from PyQt4 import QtCore, QtGui

from StereoImage import StereoImage
#from controlWindow import PreviewWindow
#from renderArea import RenderArea

class StereoArea(QtGui.QWidget):
    def __init__(self, parent=None):
        super(StereoArea, self).__init__(parent)   
        self.composer = StereoImage()
        self.imageWidth = -1
	
    def load_left(self):
        fileName = QtGui.QFileDialog.getOpenFileName(self, "Open Left Image File",
                QtCore.QDir.currentPath(), "Image files (*.tif *.jpg *.png)")
        if fileName:
            image = QtGui.QImage(fileName)
            if image.isNull():
                QtGui.QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return
                
            if image.width() > 640:
                image = image.scaledToWidth(1280)
            if image.height() > 640: 
            	image = image.scaledToHeight(720)

            self.imageLabel = image.convertToFormat(QtGui.QImage.Format_RGB32)
            self.imageWidth = image.width()
            self.imageHeight = image.height()
            
            self.composer.setSize(image.width(), image.height())
            self.composer.setLeftImage(image.bits())
            
            self.update()
            
    def load_right(self):
        fileName = QtGui.QFileDialog.getOpenFileName(self, "Open Right Image File",
                QtCore.QDir.currentPath(), "Image files (*.tif *.jpg *.png)")
        if fileName:
            image = QtGui.QImage(fileName)
            if image.isNull():
                QtGui.QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return
                
            if image.width() > 640:
                image = image.scaledToWidth(1280)
            if image.height() > 640: 
            	image = image.scaledToHeight(720)

            self.imageLabel = image.convertToFormat(QtGui.QImage.Format_RGB32)
            self.imageWidth = image.width()
            self.imageHeight = image.height()
            
            self.composer.setSize(image.width(), image.height())
            self.composer.setRightImage(image.bits())
            
            self.update()
            
    def paintEvent(self, event):
        if self.imageWidth < 0:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        painter.fillRect(event.rect(), QtGui.QBrush(QtCore.Qt.white))

        image = QtGui.QImage(self.composer.display() ,self.imageWidth, self.imageHeight, QtGui.QImage.Format_RGB32)

	painter.drawImage(0, 0, image)
	
class StereoControl(QtGui.QWidget):
    def __init__(self, parent=None):
        super(StereoControl, self).__init__(parent)  
        bigBox = QtGui.QVBoxLayout()
	
	loadBox = QtGui.QHBoxLayout()
	
	leftBtn = QtGui.QPushButton('Left')
	leftBtn.setIcon(QtGui.QIcon('./icons/unsupported.png'))
	loadBox.addWidget(leftBtn)
	
	rightBtn = QtGui.QPushButton('Right')
	rightBtn.setIcon(QtGui.QIcon('./icons/unsupported.png'))
	loadBox.addWidget(rightBtn)

	bigBox.addLayout(loadBox)
	
	self.renderView = StereoArea()
	bigBox.addWidget(self.renderView)
	
	self.setLayout(bigBox)
	
	leftBtn.clicked.connect(self.renderView.load_left)
	rightBtn.clicked.connect(self.renderView.load_right)
	

class Window(QtGui.QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        
        #self.previewWindow = PreviewWindow()

        #self.originalRenderArea = RenderArea()
        
        #self.previewWindow.valueChanged.connect(self.originalRenderArea.setXRotation)

	#self.setCentralWidget(self.originalRenderArea)
	self.setCentralWidget(StereoControl())
        
        #self.createActions()
        #self.createMenus()

        self.setWindowTitle("Transformations")

	self.resize(720, 720)

	
        
    def fitToImage(self):
    	self.resize(self.originalRenderArea.imageWidth, self.originalRenderArea.imageHeight)
    
    def showControl(self):
    	self.previewWindow.setWindowFlags(QtCore.Qt.Window)

	self.previewWindow.move(0, 24)
	self.previewWindow.show()
    	
    def createMenus(self):
        self.fileMenu = QtGui.QMenu("&File", self)
        #self.fileMenu.addAction(self.openAct)
        
        self.viewMenu = QtGui.QMenu("&View", self)
        #self.viewMenu.addAction(self.fitToImageAct)
        #self.viewMenu.addAction(self.showControlAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        
    #def createActions(self):
        #self.openAct = QtGui.QAction("&Open...", self, shortcut="Ctrl+O",
                #triggered=self.originalRenderArea.open)
        #self.fitToImageAct = QtGui.QAction("&Fit to Image", self,
                #enabled=True, checkable=False, shortcut="Ctrl+F",
                #triggered=self.fitToImage)
        #self.showControlAct = QtGui.QAction("&Control Box", self,
                #enabled=True, checkable=False, shortcut="Ctrl+U",
                #triggered=self.showControl)


if __name__ == '__main__':

    import sys

    app = QtGui.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
