/*
 * triangle asset, kd-tree, front triangulation
 */
#include <QApplication>
#include <QtCore>
#include "window.h"
#include "Parameter.h"

int main(int argc, char *argv[])
{
	tti::Parameter param(argc, argv);
	if(param.operation() == tti::Parameter::kHelp) {
		tti::Parameter::PrintHelp();
		return 0;
	}
	
    QApplication app(argc, argv);
    Window window(&param);
    //window.showMaximized();
    window.resize(800, 600);
    window.show();
    return app.exec();
}
