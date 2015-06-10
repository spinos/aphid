#include <QApplication>
#include <QDesktopWidget>
#include <QtCore>
#include "window.h"
#include "AdeniumInterface.h"
int main(int argc, char *argv[])
{
    if(argc > 1) AdeniumInterface::FileName = argv[argc - 1];
    else AdeniumInterface::FileName = "foo.hes";
    
    QApplication app(argc, argv);
    Window window;
    //window.showMaximized();
    window.resize(800, 600);
    window.show();
    return app.exec();
}
