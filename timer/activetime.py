from PyQt4 import QtCore, QtGui
import sip
import maya.OpenMayaUI as mui

RecordUIName = 'ActiveTimeRecordMayaUI'

def getMayaWindow():
    ptr = mui.MQtUtil.mainWindow()
    return sip.wrapinstance(long(ptr), QtCore.QObject)
    
def findNamedWidget(mayaName):
    wigs = QtGui.qApp.allWidgets()
    for awig in wigs:
        if awig.objectName() == mayaName:
            return awig
    return None
    
def getCurrentTimeStr():
    return QtCore.QTime.currentTime().toString('hh:mm:ss')

class MayaTimeRecordSubWindow(QtGui.QMainWindow):
    def __init__(self, parent=getMayaWindow()):
        QtGui.QMainWindow.__init__(self, parent)
        
## minimal block of time can be recorded
        self.checkPeriod = 10000
## periodically submid accumulated time
        self.breakPointThreshold = 900000
## submid once accumulated time reachs
        self.recordBlockThreshold = 300000
        self.accumTime = 0
        
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.accumActiveTime)
        timer.start(self.checkPeriod)
        timer1 = QtCore.QTimer(self)
        timer1.timeout.connect(self.breakActiveTime)
        timer1.start(self.breakPointThreshold)
        
        self.setObjectName(RecordUIName)
 
    def accumActiveTime(self):
        mnwin = self.parentWidget()
        if mnwin.isActiveWindow():
            ## print 'main is active'
            self.addTime()
        else:
            if self.discoverActiveChild(mnwin):
                ## print 'main child is active'
                self.addTime()
            else:
                ## print 'main is inactive'
                pass
                
    def breakActiveTime(self):
        if self.accumTime < self.checkPeriod:
            return
            
        self.submitRecord(self.accumTime)
        self.resetTime()
                
    def addTime(self):
        self.accumTime = self.accumTime + self.checkPeriod
        if self.accumTime >= self.recordBlockThreshold:
            self.submitRecord(self.accumTime)
            self.resetTime()
            
    def submitRecord(self, num):
## record your time here
        print 'on', getCurrentTimeStr(), 'submit', num / 60000, 'minutes for the record'

    def resetTime(self):
        self.accumTime = 0
    
    def discoverActiveChild(self, win):
        chl = win.children()
        for a in chl:
            try:
                if a.isActiveWindow():
                    return True
            except:
                pass
            
        return False
        
def LaunchMayaTimeRecord():
    if findNamedWidget(RecordUIName):
        print 'Maya Active Time Record aleady running, skip.' 
    else:
        print 'Starting Maya Active Time Record at', getCurrentTimeStr()
        MayaTimeRecordSubWindow()
        
