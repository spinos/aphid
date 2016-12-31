#include <QtGui>
#include "widget.h"
#include "window.h"
//#include "gpdfxdialog.h"

Window::Window()
{
	 qDebug()<<"NACA 4-digit airfoil";
	 QDateTime local(QDateTime::currentDateTime());
	 qDebug() << "local time is:" << local;
	 srand (local.toTime_t() );
    glWidget = new GLWidget(this);

	setCentralWidget(glWidget);
    setWindowTitle(tr("NACA 4-digit Series"));
	
	//m_xDlg = new GpdfxDialog(this);
    //m_xDlg->show();
	//m_xDlg->move(0,0);
	
	//connect(m_xDlg, SIGNAL(xValueChanged(QPointF) ), 
	//		glWidget, SLOT(recvXValue(QPointF) ) );
    
}

Window::~Window()
{ qDebug()<<"exit naca 4 window"; }

void Window::keyPressEvent(QKeyEvent *e)
{
    if (e->key() == Qt::Key_Escape)
        close();
    else
        QWidget::keyPressEvent(e);
}

