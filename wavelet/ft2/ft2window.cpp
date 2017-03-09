#include <QtGui>

#include "ft2window.h"
#include "ft2widget.h"

using namespace aphid;

Ft2Window::Ft2Window()
{
    m_plot = new Ft2Widget(this);
	
	setCentralWidget(m_plot);
    setWindowTitle(tr("2-D Discrete Wavelet Transform"));
}
//! [1]

void Ft2Window::keyPressEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Escape)
        close();

	QWidget::keyPressEvent(e);
}
