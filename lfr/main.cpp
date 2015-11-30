#include <QApplication>
#include "LfWorld.h"
#include "window.h"

using namespace lfr;

int main(int argc, char *argv[])
{
#if 1
	LfParameter param(argc, argv);
	
	if(!param.isValid()) {
		LfParameter::PrintHelp();
		return 1;
	}
	
	LfWorld world(&param);
	
    QApplication app(argc, argv);
    Window window(&world);
	//window.showMaximized();
    window.resize(800, 600);
    window.show();
    return app.exec();
#else
	LfWorld::testLAR();
    return 1;
#endif
}
