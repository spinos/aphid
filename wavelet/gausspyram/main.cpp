/*
 *   main.cpp
 *   gaussian pyramid
 */

#include <QApplication>
#include <QtCore>
#include "gauwindow.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    GauWindow window(argc, argv);
    //window.showMaximized();
    window.resize(640, 640);
    window.show();
    return app.exec();
}
