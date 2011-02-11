from PyQt4 import QtCore, QtGui


class PreviewWindow(QtGui.QWidget):
    
    valueChanged = QtCore.pyqtSignal(int)
    
    def __init__(self, parent=None):
        super(PreviewWindow, self).__init__(parent)

        self.createGridGroupBox()

        self.textEdit = QtGui.QTextEdit()
        self.textEdit.setReadOnly(True)
        self.textEdit.setLineWrapMode(QtGui.QTextEdit.NoWrap)

        closeButton = QtGui.QPushButton("&Close")
        closeButton.clicked.connect(self.close)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.gridGroupBox) 
        layout.addWidget(self.textEdit)
        layout.addWidget(closeButton)
        self.setLayout(layout)

        self.setWindowTitle("Preview")

    def createSlider(self):
	slider = QtGui.QSlider(QtCore.Qt.Horizontal)
	#slider.setFocusPolicy(QtCore.Qt.StrongFocus)
	#slider.setTickPosition(QtGui.QSlider.TicksBothSides)
	#slider.setTickInterval(10)
	slider.setSingleStep(1)
	slider.setMinimum(0)
	slider.setMaximum(255)
		
	return slider
	
    def createGridGroupBox(self):
        self.gridGroupBox = QtGui.QGroupBox("Grid layout")
        
        layout = QtGui.QGridLayout()

        
        label = QtGui.QLabel("R Scale")
        self.slider = self.createSlider()
        self.slider.valueChanged.connect(self.valueChanged)
        line = QtGui.QSpinBox()
        line.setRange(0, 255)
        layout.addWidget(label,  0, 0)
        layout.addWidget(line,  0, 1)
        layout.addWidget(self.slider,  0, 2)

        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 0)
        layout.setColumnStretch(2, 10)
        self.gridGroupBox.setLayout(layout)
        
        self.slider.valueChanged.connect(line.setValue)
        line.valueChanged.connect(self.slider.setValue)
        

    def setWindowFlags(self, flags):
        super(PreviewWindow, self).setWindowFlags(flags)

        self.textEdit.setPlainText('foo')
