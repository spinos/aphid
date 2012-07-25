from PyQt4 import QtCore, QtGui

class MainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.centralWidget = QtGui.QWidget()
        self.setCentralWidget(self.centralWidget)

        self.createImagesTable()
        
        self.createActions()
        self.createContextMenu()

        mainLayout = QtGui.QGridLayout()
        mainLayout.addWidget(self.imageTree, 1, 0)
        self.centralWidget.setLayout(mainLayout)

        self.resize(self.minimumSizeHint())

    def addImage(self):
        print 'on add image'

    def removeAllImages(self):
        print 'on remove image'

    def createImagesTable(self):
        self.imageTree = QtGui.QTreeWidget()
        self.imageTree.setHeaderLabels(['Image','Mode','Status'])

    def createActions(self):
        self.addImagesAct = QtGui.QAction("&Add Images...", self,
                shortcut="Ctrl+A", triggered=self.addImage)

        self.removeAllImagesAct = QtGui.QAction("&Remove All Images", self,
                shortcut="Ctrl+R", triggered=self.removeAllImages)

        self.exitAct = QtGui.QAction("&Quit", self, shortcut="Ctrl+Q",
                triggered=self.close)

    def createContextMenu(self):
        self.imageTree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.imageTree.customContextMenuRequested.connect(self.openMenu)
        
        #self.imageTree.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
        #self.imageTree.addAction(self.addImagesAct)
        #self.imageTree.addAction(self.removeAllImagesAct)
        #self.imageTree.addAction(self.exitAct)
        
    def openMenu(self, position):
    	menu = QtGui.QMenu()
        quitAction = menu.addAction(self.addImagesAct)
        quitAction = menu.addAction(self.removeAllImagesAct)
        quitAction = menu.addAction(self.exitAct)
        menu.exec_(self.imageTree.mapToGlobal(position))
	
if __name__ == '__main__':

    import sys

    app = QtGui.QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
