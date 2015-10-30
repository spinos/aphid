#include <QtGui>

#include "glwidget.h"
#include "window.h"

//! [0]
Window::Window()
{
    qDebug()<<"window";
    glWidget = new GLWidget;
	
	setCentralWidget(glWidget);
    setWindowTitle(tr("KdNTree"));
}
//! [1]

void Window::keyPressEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Escape)
        close();

	QWidget::keyPressEvent(e);
}
