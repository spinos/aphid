#include <QApplication>
#include <QDesktopWidget>
#include <QtCore>
#include "window.h"
#include <HesperisInterface.h>

int main(int argc, char *argv[])
{
	if(argc > 1) HesperisInterface::FileName = argv[argc - 1];
    
    qDebug()<<"main";
    QApplication app(argc, argv);
    Window window;
    //window.showMaximized();
    window.resize(800, 600);
    window.show();
    return app.exec();
}
