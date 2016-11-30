#include <QtGui>
#include "shapewidget.h"
#include "shapewindow.h"

Window::Window()
{
    glWidget = new GLWidget(this);

	setCentralWidget(glWidget);
    setWindowTitle(tr("Shape Interpolation"));
}

Window::~Window()
{ qDebug()<<" exit shape window"; }

void Window::keyPressEvent(QKeyEvent *e)
{
    if (e->key() == Qt::Key_Escape)
        close();
    else
        QWidget::keyPressEvent(e);
}

