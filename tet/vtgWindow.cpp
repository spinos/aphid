#include <QtGui>

#include "vtgWindow.h"
#include "vtgWidget.h"

namespace ttg {

Window::Window()
{
	glWidget = new vtgWidget(this);
	
	setCentralWidget(glWidget);
    setWindowTitle(tr("hexahedron grid") );
	
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