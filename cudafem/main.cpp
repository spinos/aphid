#include <QApplication>
#include <QDesktopWidget>
#include <QtCore>
#include "window.h"
#include "FemGlobal.h"
int main(int argc, char *argv[])
{
	qDebug()<<" strating cuda fem\n";
	if(argc > 1) FemGlobal::FileName = argv[argc - 1];
    else FemGlobal::FileName = "untitled.hes";
    QApplication app(argc, argv);
    Window window;
    //window.showMaximized();
    window.resize(800, 600);
    window.show();
    return app.exec();
}
