#include <QtGui>

#include "window.h"
#include "widget.h"
#include "Parameter.h"

Window::Window(const tti::Parameter * param)
{
	glWidget = new GLWidget(param->inFileName(), this);
	
	setCentralWidget(glWidget);
    setWindowTitle(tr("triangulate asset") );
	
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
