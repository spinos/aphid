#include <QtGui>

#include "window.h"
#include "haltonwidget.h"

using namespace aphid;

Window::Window()
{
    m_plot = new HaltonWidget(this);
	
	setCentralWidget(m_plot);
    setWindowTitle(tr("Halton(2,3) Sequence"));
}

void Window::keyPressEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Escape)
        close();

	QWidget::keyPressEvent(e);
}
