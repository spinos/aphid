#include <QtGui>
#include "widget.h"
#include "window.h"

Window::Window()
{
	 qDebug()<<"rotate test";
	 QDateTime local(QDateTime::currentDateTime());
	 qDebug() << "local time is:" << local;
	 srand (local.toTime_t() );
    glWidget = new GLWidget(this);

	setCentralWidget(glWidget);
    setWindowTitle(tr("Rotate Test"));
    
}

Window::~Window()
{ qDebug()<<"exit rot window"; }

void Window::keyPressEvent(QKeyEvent *e)
{
    if (e->key() == Qt::Key_Escape)
        close();
    else
        QWidget::keyPressEvent(e);
}

