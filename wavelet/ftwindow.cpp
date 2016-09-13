#include <QtGui>

#include "ftwindow.h"
#include "ft1dWidget.h"

using namespace aphid;

Window::Window()
{
    qDebug()<<"window";
    m_plot = new Ft1dWidget(this);
	
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
