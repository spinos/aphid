#include <QtGui>

#include "hegWindow.h"
#include "hegWidget.h"

namespace ttg {

Window::Window()
{
	glWidget = new hegWidget(this);
	
	setCentralWidget(glWidget);
    setWindowTitle(tr("hexahedron split") );
	
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

}