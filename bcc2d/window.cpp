#include <QtGui>
#include "glwidget.h"
#include "window.h"
#include "BccGlobal.h"
#include <HesperisFile.h>
Window::Window()
{
	glWidget = new GLWidget;

	setCentralWidget(glWidget);
	
	if(BaseFile::InvalidFilename(BccGlobal::FileName)) {
		HesperisFile hesf;
		hesf.create("untitled.hes");
		hesf.close();
		BccGlobal::FileName = "untitled.hes";
	}
	
    setWindowTitle(QString("BCC Tetrahedron Mesh - %1").arg(BccGlobal::FileName.c_str()));
}

void Window::keyPressEvent(QKeyEvent *e)
{
    if (e->key() == Qt::Key_Escape)
        close();
    else
        QWidget::keyPressEvent(e);
}
