#include <QApplication>

#include "window.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    Window window;
    //window.showMaximized();
    window.resize(640, 480);
    window.show();
    return app.exec();
}
