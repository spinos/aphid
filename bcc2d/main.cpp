#include <QApplication>
#include <QtCore>
#include <QDebug>
#include "BccGlobal.h"
#include "window.h"

//! [0]
int main(int argc, char *argv[])
{
	qDebug()<<" starting Bcc Tetrahedron\n";
	if(argc > 1) {
		BccGlobal::FileName = argv[argc-1];
	}
	
    QResource::registerResource("bcc_resource.rcc");
    QApplication app(argc, argv);
    QIcon professorIcon(":/tetrahedron.png");
    app.setWindowIcon(professorIcon);
    Window window;
    window.resize(800, 600);
	window.show();
    return app.exec();
}
//! [0]
