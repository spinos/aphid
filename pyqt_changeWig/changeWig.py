import sys, os
from PyQt4 import QtCore, QtGui

class AddressBook(QtGui.QWidget):
    def __init__(self, parent=None):
        super(AddressBook, self).__init__(parent)

        self.nameLabel = QtGui.QLabel("untitled")
        
        self.mainLayout = QtGui.QGridLayout()
        self.mainLayout.addWidget(self.nameLabel, 0, 0)
        
        self.setLayout(self.mainLayout)
        
        self.setWindowTitle("Right-click to edit the name")
        
        self.create_actions()
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.setup_context_menu)
        
    def create_actions(self):
        self.startEditAct = QtGui.QAction('Edit', self, triggered = self.start_edit_name)
        
    def setup_context_menu(self, point):
        aMenu = QtGui.QMenu()
        aMenu.addAction(self.startEditAct)
        aMenu.exec_(self.mapToGlobal(point)) 
        
    def start_edit_name(self):
        self.nameEdit = QtGui.QLineEdit(self.nameLabel.text())
        self.mainLayout.addWidget(self.nameEdit, 0, 0)
        self.nameEdit.returnPressed.connect(self.finish_edit_name)
        
    def finish_edit_name(self):
        self.nameLabel.setText(self.nameEdit.text())
        self.nameEdit.hide()

if __name__ == '__main__':
    import sys

    app = QtGui.QApplication(sys.argv)

    addressBook = AddressBook()
    addressBook.show()
    addressBook.move(100, 100)
    addressBook.resize(320, 100)

    sys.exit(app.exec_())
