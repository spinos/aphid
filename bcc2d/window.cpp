#include <QtGui>
#include "glwidget.h"
#include "window.h"
#include "BccInterface.h"
#include <HesperisFile.h>
Window::Window()
{
	glWidget = new GLWidget;

	setCentralWidget(glWidget);
	
    setWindowTitle(QString("BCC Tetrahedron Mesh - %1").arg(BccInterface::FileName.c_str()));
}

void Window::keyPressEvent(QKeyEvent *e)
{
    if (e->key() == Qt::Key_Escape)
        close();
    else
        QWidget::keyPressEvent(e);
}
