#include <QtGui>
#include "widget.h"
#include "window.h"
#include "ToolDlg.h"

Window::Window()
{
	 qDebug()<<"SLERP";
	 QDateTime local(QDateTime::currentDateTime());
	 qDebug() << "local time is:" << local;
	 srand (local.toTime_t() );
    glWidget = new GLWidget(this);
	
	ToolDlg* toolBox = new ToolDlg(this);
	
	setCentralWidget(glWidget);
    setWindowTitle(tr("SLERP"));
    
	toolBox->resize(360, 100);
	toolBox->show();
	toolBox->raise();
	toolBox->move(0,0);

	connect(toolBox, SIGNAL(toolSelected(int)),
				glWidget, SLOT(recvToolSelected(int)));
	
}

Window::~Window()
{ qDebug()<<"exit slerp window"; }

void Window::keyPressEvent(QKeyEvent *e)
{
    if (e->key() == Qt::Key_Escape)
        close();
    else
        QWidget::keyPressEvent(e);
}

