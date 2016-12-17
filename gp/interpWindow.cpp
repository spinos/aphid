#include <QtGui>
#include <iostream>
#include "interpWindow.h"
#include "interpWidget.h"

InterpWindow::InterpWindow()
{
    m_wig = new InterpWidget(this);
    setCentralWidget(m_wig);
    
    QDateTime local(QDateTime::currentDateTime());
    qDebug() << "Local time is:" << local;
    srand (local.toTime_t() );
	
	setWindowTitle(tr("GP Interpolation"));
	
}

InterpWindow::~InterpWindow()
{
	qDebug()<<"closing main window";
}

void InterpWindow::keyPressEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Escape) {
        close();
    }
	QWidget::keyPressEvent(e);
}

