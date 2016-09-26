#include <QApplication>
#include <QDesktopWidget>
#include <QtCore>
#include "dt2window.h"

int main(int argc, char *argv[])
{
    qDebug()<<"main";
    QApplication app(argc, argv);
    dt2Window window;
    //window.showMaximized();
    window.resize(640, 320);
    window.show();
    return app.exec();
}
