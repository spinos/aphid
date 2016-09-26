#include <QApplication>
#include <QDesktopWidget>
#include <QtCore>
#include "ft2window.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    Ft2Window window;
    //window.showMaximized();
    window.resize(640, 480);
    window.show();
    return app.exec();
}
