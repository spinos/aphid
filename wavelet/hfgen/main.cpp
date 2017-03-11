/*
 *   main.cpp
 *   height field generator
 */

#include <QApplication>
#include <QtCore>
#include "dthfwindow.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    DthfWindow window(argc, argv);
    //window.showMaximized();
    window.resize(640, 640);
    window.show();
    return app.exec();
}
