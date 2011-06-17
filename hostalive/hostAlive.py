import sys, os, time
from PyQt4 import QtCore, QtGui

addressArray = ['192.168.0.120','192.168.0.121','192.168.0.122','192.168.0.123','192.168.0.124']

class HostMonitor(QtGui.QTreeWidget):
    def __init__(self, parent=None):
        super(HostMonitor, self).__init__(parent)
        self.setHeaderLabels(['IP Address', ''])
        self.setColumnWidth(0, 150)
        
        for a in addressArray:
            aLeaf = QtGui.QTreeWidgetItem(self)
            aLeaf.setText(0,a)
            
    def find_host(self, address):
        for i in range(0, self.topLevelItemCount()):
            if self.topLevelItem(i).text(0) == address:
                return self.topLevelItem(i)
            
    def update(self, stat, address):
        host = self.find_host(address)
        print time.ctime()
        if stat:
            print address, 'is alive'
            host.setIcon(1,QtGui.QIcon('./greenlight.png'))
        else:
            print address, 'is dead'
            host.setIcon(1,QtGui.QIcon('./redlight.png'))
            
class TestConnect(QtCore.QThread):
    result = QtCore.pyqtSignal(bool, QtCore.QString)
    
    def __init__ (self,ip):
       super(TestConnect, self).__init__()
       self.ip = ip
       
    def run(self):
       status = True
       pingstat = os.popen('ping -q -c2 '+self.ip,'r')
       while 1:
           line = pingstat.readline()
           if not line: 
               break
           if line.find('100% packet loss') > 0:
               status = False
               
       self.result.emit(status, self.ip)

class AddressBook(QtGui.QWidget):
    def __init__(self, parent=None):
        super(AddressBook, self).__init__(parent)

        mainLayout = QtGui.QVBoxLayout()
        
        self.hosts = HostMonitor()
        
        mainLayout.addWidget(self.hosts)
        
        self.setLayout(mainLayout)
        
        self.setWindowTitle("Host Status")
        
        self.tests = []
        for a in addressArray:
            self.tests.append(TestConnect(a))
            
        for a in self.tests:
            a.result.connect(self.hosts.update)
            
        self.run_test()
            
        clock = QtCore.QTimer(self)
        clock.timeout.connect(self.run_test)
        clock.start(30000)
            
    def run_test(self):
        for a in self.tests:
            a.start()

if __name__ == '__main__':
    import sys

    app = QtGui.QApplication(sys.argv)

    addressBook = AddressBook()
    addressBook.show()
    addressBook.move(100, 100)
    addressBook.resize(320, 240)

    sys.exit(app.exec_())
