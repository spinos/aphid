#include <QApplication>
#include <QtCore>
#include "testKernelWindow.h"

using namespace aphid::gpr;

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    TestKernelWindow window;
	//window.showMaximized();
    window.resize(480, 320);
    window.show();
    return app.exec();
}
