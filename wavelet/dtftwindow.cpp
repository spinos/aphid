#include <QtGui>

#include "dtftwindow.h"
#include "dtftwidget.h"

using namespace aphid;

dtftWindow::dtftWindow()
{
    m_plot = new DtFtWidget(this);
	
	setCentralWidget(m_plot);
    setWindowTitle(tr("Dual Tree Discrete Wavelet Filters"));
}
//! [1]

void dtftWindow::keyPressEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Escape)
        close();

	QWidget::keyPressEvent(e);
}
