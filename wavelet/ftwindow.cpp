#include <QtGui>

#include "ftwindow.h"
using namespace aphid;

Window::Window()
{
    qDebug()<<"window";
    m_plot = new Plot2DWidget(this);
	
	setCentralWidget(m_plot);
    setWindowTitle(tr("Untitled"));
}
//! [1]

void Window::keyPressEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Escape)
        close();

	QWidget::keyPressEvent(e);
}
