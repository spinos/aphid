# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'foo.ui'
#
# Created: Wed Jan 26 14:01:22 2011
#      by: PyQt4 UI code generator 4.7.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui
import hoatzin

class Ui_Form(QtGui.QWidget):
    def __init__(self, parentId):      
    	Form = QtGui.QWidget.find(parentId)
        super(Ui_Form, self).__init__(Form)
        self.resize(400, 100)
        self.host = hoatzin.Hoatzin()
        self.boxLayout = QtGui.QHBoxLayout()
        self.boxLayout.setObjectName('ui_form_box')
        self.boxLayout.setObjectName("boxLayout")
        self.boxLayout.addWidget(QtGui.QLabel('pyqt widget'))
        self.dial = QtGui.QDial()
        self.boxLayout.addWidget(self.dial)
        
        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.boxLayout.addWidget(self.slider)
        
        self.setLayout(self.boxLayout)

        self.dial.valueChanged.connect(self.dialTest)
        self.slider.valueChanged.connect(self.sliderTest)

   
    def dialTest(self):
        w = self.host.set_attribute_int('SnakeHole', 'fieldNumber', self.dial.value() )
        if w != 1:
            print 'SnakeHole not found'
        
    def sliderTest(self):
        w = self.host.set_attribute_int('SnakeHole', 'sliderNumber', self.slider.value() )
        if w != 1:
            print 'SnakeHole not found'     
    
