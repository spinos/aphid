/*
 *  garden main entry
 */
#include <QApplication>
#include <QtCore>
#include "ui/window.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    Window window;
    //window.showMaximized();
    //window.resize(800, 600);
    window.show();
	window.showDlgs();
    return app.exec();
}
