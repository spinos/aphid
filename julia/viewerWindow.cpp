#include <QtGui>

#include "viewerWindow.h"
#include "whitenoisewidget.h"
#include "CubeRender.h"
#include "WorldRender.h"

namespace jul {

Window::Window(aphid::CudaRender * r,
				const std::string & title)
{
    m_widget = new MandelbrotWidget(r, this);
	
	setCentralWidget(m_widget);
	setWindowTitle(tr(title.c_str() ) );
}
//! [1]

void Window::keyPressEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Escape)
        close();

	QWidget::keyPressEvent(e);
}

}
