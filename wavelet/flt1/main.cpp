#include <QApplication>
#include <QDesktopWidget>
#include <QtCore>
#include "ftwindow.h"

int main(int argc, char *argv[])
{
    qDebug()<<"main";
    QApplication app(argc, argv);
    Window window;
    //window.showMaximized();
    window.resize(400, 300);
    window.show();
    return app.exec();
}
