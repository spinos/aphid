#include <QApplication>
#include <QDesktopWidget>
#include <QtCore>
#include "window.h"
#include "FemGlobal.h"
#include <iostream>
int main(int argc, char *argv[])
{
	std::cout<<"Cuda FEM start up\n";
	if(argc > 1) FemGlobal::FileName = argv[argc - 1];
    else FemGlobal::FileName = "untitled.hes";
// external binary resource
// rcc -binary image.qrc -o resource.rcc
    QResource::registerResource("resource.rcc");
    QApplication app(argc, argv);
    QIcon professorIcon(":/professor.png");
    app.setWindowIcon(professorIcon);
    Window window;
    //window.showMaximized();
    window.resize(800, 600);
    window.show();
    return app.exec();
}
