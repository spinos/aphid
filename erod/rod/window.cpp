/*
 *  rodt
 */
#include <QtGui>

#include "glwidget.h"
#include "window.h"

Window::Window()
{
    glWidget = new GLWidget;
	
	setCentralWidget(glWidget);
    setWindowTitle(tr("Position Based Elastic Rod"));
}

void Window::keyPressEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Escape)
        close();

	QWidget::keyPressEvent(e);
}
