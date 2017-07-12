#include <QtGui>
#include "widget.h"
#include "window.h"

Window::Window()
{
    glWidget = new GLWidget(this);

	setCentralWidget(glWidget);
    setWindowTitle(tr("Instance Draw + Energy based Particle"));
}

Window::~Window()
{ qDebug()<<" exit instdr window"; }

void Window::keyPressEvent(QKeyEvent *e)
{
    if (e->key() == Qt::Key_Escape)
        close();
    else
        QWidget::keyPressEvent(e);
}

