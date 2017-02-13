#include <QtGui>
#include "widget.h"
#include "window.h"
#include "Parameter.h"

Window::Window(const exrs::Parameter * param)
{
	 qDebug()<<"test exr sampler";
	 QDateTime local(QDateTime::currentDateTime());
	 qDebug() << "local time is:" << local;
	 srand (local.toTime_t() );
    glWidget = new GLWidget(param->inFileName(), this);

	setCentralWidget(glWidget);
    setWindowTitle(tr("EXR Sampler Test"));
    
}

Window::~Window()
{ qDebug()<<"exit exrs window"; }

void Window::keyPressEvent(QKeyEvent *e)
{
    if (e->key() == Qt::Key_Escape)
        close();
    else
        QWidget::keyPressEvent(e);
}

