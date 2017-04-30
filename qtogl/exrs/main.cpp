/// test exr sampler
#include <QApplication>

#include "window.h"
#include "Parameter.h"

int main(int argc, char *argv[])
{
	exrs::Parameter param(argc, argv);
	if(param.operation() == exrs::Parameter::kHelp) {
		exrs::Parameter::PrintHelp();
		return 0;
	}
	
    QApplication app(argc, argv);
    Window window(&param);
    //window.showMaximized();
    window.resize(800, 600);
    window.show();
    return app.exec();
}
