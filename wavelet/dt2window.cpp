#include <QtGui>

#include "dt2window.h"
#include "dt2widget.h"

using namespace aphid;

dt2Window::dt2Window()
{
    m_plot = new Dt2Widget(this);
	
	setCentralWidget(m_plot);
    setWindowTitle(tr("2-D Dual Tree Discrete Wavelet Transform"));
}
//! [1]

void dt2Window::keyPressEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Escape)
        close();

	QWidget::keyPressEvent(e);
}
