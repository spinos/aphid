#include <QtGui>

#include "viewerWindow.h"
#include "whitenoisewidget.h"
#include "CubeRender.h"
#include "WorldRender.h"

namespace jul {

Window::Window(const Parameter * param)
{
	aphid::CudaRender * r;
	
	if(param->operation() == Parameter::kView) {
		r = new aphid::WorldRender(param->outFileName() );
		setWindowTitle(tr(param->outFileName().c_str()));
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
