
from PyQt4 import QtCore, QtGui


class AddressBook(QtGui.QWidget):
    def __init__(self, parent=None):
        super(AddressBook, self).__init__(parent)

        nameLabel = QtGui.QLabel("Note:")
        #self.nameLine = QtGui.QLineEdit()

        self.addressText = QtGui.QPushButton("Click to Change Color")
        self.addressText.setStyleSheet("QPushButton {background-color: darkkhaki; border-style: solid; border-radius: 5;} QPushButton:checked { background-color: green;}")
        self.addressText.setCheckable(1)
        self.addressText.setChecked(0)
        mainLayout = QtGui.QGridLayout()

        mainLayout.addWidget( nameLabel, 0, 0)
        mainLayout.addWidget(self.addressText, 0, 1)

        self.setLayout(mainLayout)
        self.setWindowTitle("Test")


if __name__ == '__main__':
    import sys

    app = QtGui.QApplication(sys.argv)

    addressBook = AddressBook()
    addressBook.show()

    sys.exit(app.exec_())
