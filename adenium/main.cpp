#include <QApplication>
#include <QDesktopWidget>
#include <QtCore>
#include "window.h"
#include "AdeniumInterface.h"
#include <iostream>
int main(int argc, char *argv[])
{
    if(argc > 1) AdeniumInterface::FileName = argv[argc - 1];
    else AdeniumInterface::FileName = "foo.hes";
    
    std::cout<<"Adenium start up"
    <<"\n press Q to switch between ray-casting and openGL "
    <<"\n press L to load bake file"
    <<"\n ";
    
    QApplication app(argc, argv);
    Window window;
    //window.showMaximized();
    window.resize(800, 600);
    window.show();
    return app.exec();
}
