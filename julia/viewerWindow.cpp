#include <QtGui>

#include "viewerWindow.h"
#include "whitenoisewidget.h"
#include "CubeRender.h"
#include "WorldRender.h"

namespace jul {

Window::Window(int argc, char *argv[])
{
	aphid::CudaRender * r;
	
	if(argc == 2) {
		r = new aphid::WorldRender(argv[1] );
		setWindowTitle(tr(argv[1]));
	}
	else {
		r = new aphid::CubeRender;
		setWindowTitle(tr("ray-cast test"));
	}
		
    m_widget = new MandelbrotWidget(r, this);
	
	setCentralWidget(m_widget);
}
//! [1]

void Window::keyPressEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Escape)
        close();

	QWidget::keyPressEvent(e);
}

}
