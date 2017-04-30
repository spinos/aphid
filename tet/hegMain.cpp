#include <QApplication>
#include <QDesktopWidget>
#include <QtCore>
#include "hegWindow.h"

using namespace ttg;
 
int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    Window window;
    //window.showMaximized();
    window.resize(800, 600);
    window.show();
    return app.exec();
}
