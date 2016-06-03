#include <QApplication>
#include <QDesktopWidget>
#include <QtCore>
#include "window.h"

using namespace ttg;
 
int main(int argc, char *argv[])
{
	Parameter param(argc, argv);
	if(param.operation() == Parameter::kHelp ) {
		Parameter::PrintHelp();
		return 1;
	}
	
    QApplication app(argc, argv);
    Window window(&param);
    //window.showMaximized();
    window.resize(800, 600);
    window.show();
    return app.exec();
}
