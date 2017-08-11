#include <QtGui>
#include "window.h"
#include "RenderWidget.h"

Window::Window()
{
    m_view = new RenderWidget(this);
	setCentralWidget(m_view);
    setWindowTitle(tr("RTRT"));
}

void Window::keyPressEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Escape)
        close();

	QWidget::keyPressEvent(e);
}
