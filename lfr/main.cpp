#include <QApplication>
#include "LfWorld.h"
#include "LfWidget.h"
#include <iostream>

int main(int argc, char *argv[])
{
	LfParameter param(argc, argv);
	
	if(!param.isValid()) {
		LfParameter::PrintHelp();
		return 1;
	}
	
	LfWorld world(param);
	
    QApplication app(argc, argv);
    LfWidget widget(&world);
    widget.show();
    return app.exec();
}
