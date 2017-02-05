#include <QtGui>

#include "window.h"
#include "widget.h"

Window::Window()
{
	glWidget = new GLWidget(this);
	
	setCentralWidget(glWidget);
    setWindowTitle(tr("adaptive bcc grid") );
	
	createActions();
    createMenus();
	
}

void Window::keyPressEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Escape)
        close();

	QWidget::keyPressEvent(e);
}

void Window::createActions()
{

}
	
void Window::createMenus()
{

}
