import sys, os
from PyQt4 import QtCore, QtGui
import random

class AddressBook(QtGui.QWidget):
    def __init__(self, parent=None):
        super(AddressBook, self).__init__(parent)
        
        mainLayout = QtGui.QHBoxLayout()
        
        label = QtGui.QLabel("Menu")
        label.setFrameStyle(label.Box)
        label.setMinimumWidth(100)
        
        policy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Expanding)
        label.setSizePolicy(policy)
        mainLayout.addWidget(label)
        

        self.styleComboBox = QtGui.QComboBox()
        self.styleComboBox.addItem("abc________")
        self.styleComboBox.setSizeAdjustPolicy(QtGui.QComboBox.AdjustToContents) 
        mainLayout.addWidget(self.styleComboBox)
        
        mainLayout.addStretch(1)
        self.setLayout(mainLayout)

        self.setWindowTitle("Combo Sizing Policy")
        
        self.styleComboBox.activated[str].connect(self.addStyle)
        self.acc = 'sos'
        
    def addStyle(self, stl):
        self.acc = '%sx%x' % (self.acc, random.randint(10, 10000))
        self.styleComboBox.addItem(self.acc)
        
if __name__ == '__main__':
    import sys

    app = QtGui.QApplication(sys.argv)

    addressBook = AddressBook()
    
    addressBook.setGeometry(100, 100, 640, 480)
    addressBook.show()

    sys.exit(app.exec_())
    
