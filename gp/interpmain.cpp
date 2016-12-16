#include <QApplication>

#include "interpWindow.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    InterpWindow window;
    //window.showMaximized();
    window.resize(800, 600);
    window.show();
    return app.exec();
}

