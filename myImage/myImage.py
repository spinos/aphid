#
# Simple Image Viewer using QImage
#

from PyQt4 import QtCore, QtGui

from controlWindow import PreviewWindow
from renderArea import RenderArea
        
    

class Window(QtGui.QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        
        self.previewWindow = PreviewWindow()

        self.originalRenderArea = RenderArea()
        
        self.previewWindow.valueChanged.connect(self.originalRenderArea.setXRotation)

	self.setCentralWidget(self.originalRenderArea)
        
        self.createActions()
        self.createMenus()

        self.setWindowTitle("Transformations")

	self.resize(1280, 720)
        
    def fitToImage(self):
    	self.resize(self.originalRenderArea.imageWidth, self.originalRenderArea.imageHeight)
    
    def showControl(self):
    	self.previewWindow.setWindowFlags(QtCore.Qt.Window)

	self.previewWindow.move(0, 24)
	self.previewWindow.show()
    	
    def createMenus(self):
        self.fileMenu = QtGui.QMenu("&File", self)
        self.fileMenu.addAction(self.openAct)
        
        self.viewMenu = QtGui.QMenu("&View", self)
        self.viewMenu.addAction(self.fitToImageAct)
        self.viewMenu.addAction(self.showControlAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        
    def createActions(self):
        self.openAct = QtGui.QAction("&Open...", self, shortcut="Ctrl+O",
                triggered=self.originalRenderArea.open)
        self.fitToImageAct = QtGui.QAction("&Fit to Image", self,
                enabled=True, checkable=False, shortcut="Ctrl+F",
                triggered=self.fitToImage)
        self.showControlAct = QtGui.QAction("&Control Box", self,
                enabled=True, checkable=False, shortcut="Ctrl+U",
                triggered=self.showControl)


if __name__ == '__main__':

    import sys

    app = QtGui.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
