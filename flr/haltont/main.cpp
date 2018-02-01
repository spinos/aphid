#include <QApplication>
#include <QDesktopWidget>
#include <QtCore>
#include "window.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    Window window;
    //window.showMaximized();
    window.resize(480, 480);
    window.show();
    return app.exec();
}
