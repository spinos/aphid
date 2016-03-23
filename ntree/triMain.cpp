#include <QApplication>
#include <QDesktopWidget>
#include <QtCore>
#include "triWindow.h"

int main(int argc, char *argv[])
{
	std::string tgt;
    if(argc>1) tgt = argv[1];
    QApplication app(argc, argv);
    TriWindow window(tgt);
    //window.showMaximized();
    window.resize(720, 540);
    window.show();
    return app.exec();
}
