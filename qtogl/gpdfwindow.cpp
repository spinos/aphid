#include <QtGui>
#include "gpdfwidget.h"
#include "gpdfwindow.h"
#include "gpdfxdialog.h"

Window::Window()
{
	 qDebug()<<"gp deformation";
	 QDateTime local(QDateTime::currentDateTime());
	 qDebug() << "local time is:" << local;
	 srand (local.toTime_t() );
    glWidget = new GLWidget(this);

	setCentralWidget(glWidget);
    setWindowTitle(tr("GP interpolation N dimension"));
	
	m_xDlg = new GpdfxDialog(this);
    m_xDlg->show();
	m_xDlg->move(0,0);
	
	connect(m_xDlg, SIGNAL(xValueChanged(QPointF) ), 
			glWidget, SLOT(recvXValue(QPointF) ) );
    
}

Window::~Window()
{ qDebug()<<"exit gp deformation window"; }

void Window::keyPressEvent(QKeyEvent *e)
{
    if (e->key() == Qt::Key_Escape)
        close();
    else
        QWidget::keyPressEvent(e);
}

