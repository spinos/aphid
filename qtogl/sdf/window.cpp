#include <QtGui>
#include "widget.h"
#include "window.h"
#include <qt/SuperformulaControl.h>

using namespace aphid;

Window::Window()
{
	 qDebug()<<"sdf test";
	 QDateTime local(QDateTime::currentDateTime());
	 qDebug() << "local time is:" << local;
	 srand (local.toTime_t() );
    glWidget = new GLWidget(this);
	m_superformulaControl = new SuperformulaControl(this);

	setCentralWidget(glWidget);
    setWindowTitle(tr("SDF Test"));
    
	connect(m_superformulaControl, SIGNAL(a1Changed(double) ), 
			glWidget, SLOT(receiveA1(double)));
			
	connect(m_superformulaControl, SIGNAL(b1Changed(double) ), 
		glWidget, SLOT(receiveB1(double)));
		
	connect(m_superformulaControl, SIGNAL(m1Changed(double) ), 
		glWidget, SLOT(receiveM1(double)));
		
	connect(m_superformulaControl, SIGNAL(n1Changed(double) ), 
		glWidget, SLOT(receiveN1(double)));
		
	connect(m_superformulaControl, SIGNAL(n2Changed(double) ), 
		glWidget, SLOT(receiveN2(double)));
		
	connect(m_superformulaControl, SIGNAL(n3Changed(double) ), 
		glWidget, SLOT(receiveN3(double)));
		
	connect(m_superformulaControl, SIGNAL(a2Changed(double) ), 
		glWidget, SLOT(receiveA2(double)));
		
	connect(m_superformulaControl, SIGNAL(b2Changed(double) ), 
		glWidget, SLOT(receiveB2(double)));
		
	connect(m_superformulaControl, SIGNAL(m2Changed(double) ), 
		glWidget, SLOT(receiveM2(double)));
		
	connect(m_superformulaControl, SIGNAL(n21Changed(double) ), 
		glWidget, SLOT(receiveN21(double)));
		
	connect(m_superformulaControl, SIGNAL(n22Changed(double) ), 
		glWidget, SLOT(receiveN22(double)));
		
	connect(m_superformulaControl, SIGNAL(n23Changed(double) ), 
		glWidget, SLOT(receiveN23(double)));
			
	m_superformulaControl->show();
	
}

Window::~Window()
{ qDebug()<<"exit legendre sdf window"; }

void Window::keyPressEvent(QKeyEvent *e)
{
    if (e->key() == Qt::Key_Escape)
        close();
    else
        QWidget::keyPressEvent(e);
}

