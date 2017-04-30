#include <QApplication>
#include <QDesktopWidget>
#include <FemGlobal.h>
#include <qDebug>
#include "window.h"

int main(int argc, char *argv[])
{
	qDebug()<<" starting Fem\n";
	if(argc > 1)
		FemGlobal::FileName = argv[argc-1];
	
    QApplication app(argc, argv);
    Window window;
    //window.showMaximized();
    window.resize(800, 600);
    window.show();
    return app.exec();
}
