#include <QtGui>

#include "glwidget.h"
#include "window.h"

//! [0]
Window::Window()
{
    glWidget = new GLWidget;

	setCentralWidget(glWidget);
    setWindowTitle(tr("Convex Hull Test"));
}

void Window::keyPressEvent(QKeyEvent *e)
{
    if (e->key() == Qt::Key_Escape)
        close();
    else
        QWidget::keyPressEvent(e);
}
