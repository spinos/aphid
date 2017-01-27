/*
 * tetrahedron grid
 * http://people.sc.fsu.edu/~jburkardt/cpp_src/tetrahedron_grid/tetrahedron_grid.html
 */
#include <QApplication>
#include <QtCore>
#include "window.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    Window window;
    //window.showMaximized();
    window.resize(800, 600);
    window.show();
    return app.exec();
}
