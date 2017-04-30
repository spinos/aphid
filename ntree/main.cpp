#include <QApplication>
#include <QDesktopWidget>
#include <QtCore>
#include "window.h"

int main(int argc, char *argv[])
{
	QApplication app(argc, argv);
    Window window(argc, argv);
    //window.showMaximized();
    window.resize(720, 540);
    window.show();
    return app.exec();
}
