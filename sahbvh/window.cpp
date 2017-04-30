#include <QtGui>

#include "glwidget.h"
#include "window.h"

//! [0]
Window::Window()
{
    glWidget = new GLWidget;
	
	setCentralWidget(glWidget);
    setWindowTitle(tr("Cuda SAH BVH"));
}
//! [1]

void Window::keyPressEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Escape)
        close();

	QWidget::keyPressEvent(e);
}
