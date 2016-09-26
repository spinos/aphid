#include <QApplication>
#include <QDesktopWidget>
#include <QtCore>
#include "dtftwindow.h"

int main(int argc, char *argv[])
{
    qDebug()<<"main";
    QApplication app(argc, argv);
    dtftWindow window;
    //window.showMaximized();
    window.resize(400, 300);
    window.show();
    return app.exec();
}
