#include <QtGui>
#include "widget.h"
#include "window.h"

Window::Window()
{
    glWidget = new GLWidget(this);

	setCentralWidget(glWidget);
    setWindowTitle(tr("Real Instance Draw"));
}

Window::~Window()
{ qDebug()<<" exit instr window"; }

void Window::keyPressEvent(QKeyEvent *e)
{
    if (e->key() == Qt::Key_Escape)
        close();
    else
        QWidget::keyPressEvent(e);
}

