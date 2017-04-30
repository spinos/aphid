#include <QApplication>
#include <QDesktopWidget>
#include <QtCore>
#include "window.h"
#include "SahGlobal.h"
int main(int argc, char *argv[])
{
	qDebug()<<" strating sah bvh test\n";
	if(argc > 1) SahGlobal::FileName = argv[argc - 1];
    else SahGlobal::FileName = "../bcc2d/untitled.hes";
    QApplication app(argc, argv);
    Window window;
    //window.showMaximized();
    window.resize(800, 600);
    window.show();
    return app.exec();
}
