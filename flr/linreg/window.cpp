#include <QtGui>

#include "window.h"
#include "linregwidget.h"

using namespace aphid;

Window::Window()
{
    m_plot = new LinregWidget(this);
	
	setCentralWidget(m_plot);
    setWindowTitle(tr("Linear Model Regression"));
}

void Window::keyPressEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Escape)
        close();

	QWidget::keyPressEvent(e);
}
