#include <QtGui>
#include "shapewidget.h"
#include "shapewindow.h"
#include "viewKernelDialog.h"

using namespace aphid;

Window::Window()
{
    glWidget = new GLWidget(this);
	
	m_kernDlg = new gpr::ViewKernelDialog(&glWidget->K(), this);
    
	setCentralWidget(glWidget);
    setWindowTitle(tr("Shape Interpolation"));
	
	m_kernDlg->show();
	m_kernDlg->move(0,0);
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

