#include <QtGui>

#include "vdfWindow.h"
#include "vdfScene.h"
#include "vdfWidget.h"

namespace ttg {

Window::Window(const Parameter * param)
{
	Scene * sc = new vdfScene(param->inFileName() );
	
    glWidget = new vdfWidget(sc, this);
	
	setCentralWidget(glWidget);
    setWindowTitle(tr(sc->titleStr() ) );
	
	createActions(param->operation() );
    createMenus(param->operation() );
	
}

void Window::keyPressEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Escape)
        close();

	QWidget::keyPressEvent(e);
}

void Window::createActions(Parameter::Operation opt)
{

}
	
void Window::createMenus(Parameter::Operation opt)
{

}

}