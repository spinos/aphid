import sys, os
from PyQt4 import QtCore, QtGui

class AddressBook(QtGui.QWidget):
    def __init__(self, parent=None):
        super(AddressBook, self).__init__(parent)
        
        self.create_actions()

        mainLayout = QtGui.QVBoxLayout()
        
        toolBar = QtGui.QToolBar()
        
        simpleButton = QtGui.QToolButton()
        simpleButton.setDefaultAction(self.simpleAct)
        fileButton = QtGui.QToolButton()
        fileButton.setText('File')
        fileButton.setPopupMode(QtGui.QToolButton.MenuButtonPopup)
        fileButton.setMenu(self.pop_file_menu())
        editButton = QtGui.QToolButton()
        editButton.setText('Edit')
        editButton.setPopupMode(QtGui.QToolButton.MenuButtonPopup)
        editButton.setMenu(self.pop_edit_menu())
        
        toolBar.addWidget(simpleButton)
        toolBar.addWidget(fileButton)
        toolBar.addWidget(editButton)
        
        mainLayout.addWidget(toolBar)
        mainLayout.addWidget(QtGui.QGroupBox())
        
        self.setLayout(mainLayout)

        self.setWindowTitle("Menu By Tool Bar")
        
    def pop_file_menu(self):
        
        aMenu = QtGui.QMenu(self)
        aMenu.addAction(self.fileOpenAct)
        aMenu.addAction(self.fileCloseAct)
        return aMenu
        
    def pop_edit_menu(self):
        
        aMenu = QtGui.QMenu(self)
        aMenu.addAction(self.editCopyAct)
        aMenu.addAction(self.filePasteAct)
        return aMenu
        
    def create_actions(self):
        self.simpleAct = QtGui.QAction('Simple', self, triggered=self.do_nothing)
        self.fileOpenAct = QtGui.QAction('Open', self, triggered=self.do_nothing)
        self.fileCloseAct = QtGui.QAction('Close', self, triggered=self.do_nothing)
        self.editCopyAct = QtGui.QAction('Copy', self, triggered=self.do_nothing)
        self.filePasteAct = QtGui.QAction('Paste', self, triggered=self.do_nothing)
    
    def do_nothing(self):
        print 'do nothing'

if __name__ == '__main__':
    import sys

    app = QtGui.QApplication(sys.argv)

    addressBook = AddressBook()
    
    addressBook.setGeometry(100, 100, 640, 480)
    addressBook.show()

    sys.exit(app.exec_())
    
