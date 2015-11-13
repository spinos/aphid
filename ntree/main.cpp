#include <QApplication>
#include <QDesktopWidget>
#include <QtCore>
#include "window.h"

int main(int argc, char *argv[])
{
    qDebug()<<"main";
    QApplication app(argc, argv);
    Window window;
    //window.showMaximized();
    window.resize(640, 640);
    window.show();
    return app.exec();
}
