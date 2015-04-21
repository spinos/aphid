#include <QtGui>
#include "glwidget.h"
#include "window.h"
#include "BccGlobal.h"
Window::Window()
{
	glWidget = new GLWidget;

	setCentralWidget(glWidget);
    setWindowTitle(tr("BCC Tetrahedron Mesh"));
}

void Window::keyPressEvent(QKeyEvent *e)
{
    if (e->key() == Qt::Key_Escape)
        close();
    else
        QWidget::keyPressEvent(e);
}
