#include <QtGui>

#include "viewerWindow.h"
#include "whitenoisewidget.h"

namespace jul {

Window::Window(const Parameter * param)
{
	CubeRender * r = new CubeRender;
	
    m_widget = new MandelbrotWidget(r, this);
	
	setCentralWidget(m_widget);
    
	if(param->operation() == Parameter::kView) 
		setWindowTitle(tr(param->outFileName().c_str()));
	else 
		setWindowTitle(tr("ray cast test"));
}
//! [1]

void Window::keyPressEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Escape)
        close();

	QWidget::keyPressEvent(e);
}

}
