#!/usr/bin/env python


#############################################################################
##
## Copyright (C) 2010 Riverbank Computing Limited.
## Copyright (C) 2010 Nokia Corporation and/or its subsidiary(-ies).
## All rights reserved.
##
## This file is part of the examples of PyQt.
##
## $QT_BEGIN_LICENSE:BSD$
## You may use this file under the terms of the BSD license as follows:
##
## "Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are
## met:
##   * Redistributions of source code must retain the above copyright
##     notice, this list of conditions and the following disclaimer.
##   * Redistributions in binary form must reproduce the above copyright
##     notice, this list of conditions and the following disclaimer in
##     the documentation and/or other materials provided with the
##     distribution.
##   * Neither the name of Nokia Corporation and its Subsidiary(-ies) nor
##     the names of its contributors may be used to endorse or promote
##     products derived from this software without specific prior written
##     permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
## "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
## LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
## A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
## OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
## SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
## LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
## DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
## THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
## (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
## $QT_END_LICENSE$
##
#############################################################################


# This is only needed for Python v2 but is harmless for Python v3.
import sip
sip.setapi('QVariant', 2)

from math import cos, pi, sin

from PyQt4 import QtCore, QtGui


class RenderArea(QtGui.QWidget):
    def __init__(self, imageFilename, parent=None):
        super(RenderArea, self).__init__(parent)

        self.imageFilename = imageFilename
        self.penWidth = 1
        self.rotationAngle = 0
        self.setBackgroundRole(QtGui.QPalette.Base)

    def minimumSizeHint(self):
        return QtCore.QSize(50, 50)

    def sizeHint(self):
        return QtCore.QSize(100, 100)

    def paintEvent(self, event):
        
        picture = QtGui.QPixmap()
        picture.load(self.imageFilename)
        picture = picture.scaled(100, 100, QtCore.Qt.IgnoreAspectRatio)
        
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.scale(self.width() / 100.0, self.height() / 100.0)
        
        painter.drawPixmap(0, 0, picture)


class Window(QtGui.QWidget):
    NumRenderAreas = 9

    def __init__(self):
        super(Window, self).__init__()


        ellipsePath = QtGui.QPainterPath()
        ellipsePath.moveTo(80.0, 50.0)
        ellipsePath.arcTo(20.0, 30.0, 60.0, 40.0, 0.0, 360.0)

        self.renderAreas = RenderArea('C:\\Users\\zhangjian\\Pictures\\27029_single.jpg')


        self.fillRuleComboBox = QtGui.QComboBox()
        self.fillRuleComboBox.addItem("Odd Even", QtCore.Qt.OddEvenFill)
        self.fillRuleComboBox.addItem("Winding", QtCore.Qt.WindingFill)

        fillRuleLabel = QtGui.QLabel("Fill &Rule:")
        fillRuleLabel.setBuddy(self.fillRuleComboBox)

        self.fillColor1ComboBox = QtGui.QComboBox()
        
        self.fillColor1ComboBox.setCurrentIndex(
                self.fillColor1ComboBox.findText("mediumslateblue"))

        self.fillColor2ComboBox = QtGui.QComboBox()
        
        self.fillColor2ComboBox.setCurrentIndex(
                self.fillColor2ComboBox.findText("cornsilk"))

        fillGradientLabel = QtGui.QLabel("&Fill Gradient:")
        fillGradientLabel.setBuddy(self.fillColor1ComboBox)

        fillToLabel = QtGui.QLabel("to")
        fillToLabel.setSizePolicy(QtGui.QSizePolicy.Fixed,
                QtGui.QSizePolicy.Fixed)

        self.penWidthSpinBox = QtGui.QSpinBox()
        self.penWidthSpinBox.setRange(0, 20)

        penWidthLabel = QtGui.QLabel("&Pen Width:")
        penWidthLabel.setBuddy(self.penWidthSpinBox)

        self.penColorComboBox = QtGui.QComboBox()
        
        self.penColorComboBox.setCurrentIndex(
                self.penColorComboBox.findText('darkslateblue'))

        penColorLabel = QtGui.QLabel("Pen &Color:")
        penColorLabel.setBuddy(self.penColorComboBox)

        self.rotationAngleSpinBox = QtGui.QSpinBox()
        self.rotationAngleSpinBox.setRange(0, 359)
        self.rotationAngleSpinBox.setWrapping(True)
        self.rotationAngleSpinBox.setSuffix('\xB0')

        rotationAngleLabel = QtGui.QLabel("&Rotation Angle:")
        rotationAngleLabel.setBuddy(self.rotationAngleSpinBox)

        
        mainLayout = QtGui.QGridLayout()
        mainLayout.addWidget(self.renderAreas, 0, 0)
        
        self.setLayout(mainLayout)

        self.setWindowTitle("Painter Paths")

    

if __name__ == '__main__':

    import sys

    app = QtGui.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
