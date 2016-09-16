#include <QtGui>

#include "dtftwindow.h"
#include "dtftwidget.h"

using namespace aphid;

dtftWindow::dtftWindow()
{
    m_plot = new DtFtWidget(this);
	
	setCentralWidget(m_plot);
    setWindowTitle(tr("1-D Dual Tree Discrete Wavelet Transform"));
}
//! [1]

void dtftWindow::keyPressEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Escape)
        close();

	QWidget::keyPressEvent(e);
}
