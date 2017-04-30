#include <QtGui>

#include "window.h"
#include "widget.h"

Window::Window()
{
	QDateTime local(QDateTime::currentDateTime());
	 qDebug() << "local time is:" << local;
	 srand (local.toTime_t() );
	 
	glWidget = new GLWidget(this);
	
	setCentralWidget(glWidget);
    setWindowTitle(tr("kmean clustering test") );
	
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
