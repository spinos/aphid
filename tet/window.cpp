#include <QtGui>

#include "glwidget.h"
#include "window.h"
#include "Delaunay2D.h"
#include "Delaunay3D.h"
#include "Hilbert2D.h"
#include "Hilbert3D.h"
#include "Bcc3dTest.h"
#include "SuperformulaTest.h"
#include "SuperformulaPoisson.h"
#include "SuperformulaControl.h"
#include "BccTetrahedralize.h"
#include "DistanceFieldTest.h"
#include "RedBlueTest.h"
#include "RedBlueControl.h"

namespace ttg {

Window::Window(const Parameter * param)
{
	Scene * sc = NULL;
	if(param->operation() == Parameter::kHilbert2D)
		sc = new Hilbert2D;
	else if(param->operation() == Parameter::kHilbert3D)
		sc = new Hilbert3D;
	else if(param->operation() == Parameter::kDelaunay3D)
		sc = new Delaunay3D;
	else if(param->operation() == Parameter::kBcc3D)
		sc = new Bcc3dTest;
	else if(param->operation() == Parameter::kDelaunay2D)
		sc = new Delaunay2D;
	else if(param->operation() == Parameter::kSuperformula)
		sc = new SuperformulaTest;
	else if(param->operation() == Parameter::kSuperformulaPoissonDisk)
		sc = new SuperformulaPoisson;
	else if(param->operation() == Parameter::kBccTetrahedralize)
		sc = new BccTetrahedralize;
	else if(param->operation() == Parameter::kDistanceField)
		sc = new DistanceFieldTest;
	else// if(param->operation() == Parameter::kRedblueRefine)
		sc = new RedBlueTest;
		
    glWidget = new GLWidget(sc, this);
	
	setCentralWidget(glWidget);
    setWindowTitle(tr(sc->titleStr() ) );
	
	createActions(param->operation() );
    createMenus(param->operation() );
	
}
//! [1]

void Window::keyPressEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Escape)
        close();

	QWidget::keyPressEvent(e);
}

void Window::createActions(Parameter::Operation opt)
{
	if(opt == Parameter::kSuperformula
		|| opt == Parameter::kSuperformulaPoissonDisk
		|| opt == Parameter::kBccTetrahedralize ) {
		m_superformulaControl = new SuperformulaControl(this);
	
		showSFControlAct = new QAction(tr("&Superformula Control"), this);
		showSFControlAct->setStatusTip(tr("Show Superformula settings"));
		connect(showSFControlAct, SIGNAL(triggered()), m_superformulaControl, SLOT(show()));
    
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
			
	}
	else if(opt == Parameter::kRedblueRefine) {
		m_redblueControl = new RedBlueControl(this);
		showRBControlAct = new QAction(tr("&Red Blue Refine Control"), this);
		showRBControlAct->setStatusTip(tr("Show Red Blue settings"));
		connect(showRBControlAct, SIGNAL(triggered()), m_redblueControl, SLOT(show()));
    
		connect(m_redblueControl, SIGNAL(aChanged(double) ), 
			glWidget, SLOT(receiveA(double)));
			
		connect(m_redblueControl, SIGNAL(bChanged(double) ), 
			glWidget, SLOT(receiveB(double)));
			
		connect(m_redblueControl, SIGNAL(cChanged(double) ), 
			glWidget, SLOT(receiveC(double)));
			
		connect(m_redblueControl, SIGNAL(dChanged(double) ), 
			glWidget, SLOT(receiveD(double)));
		
	}
}
	
void Window::createMenus(Parameter::Operation opt)
{
	if(opt == Parameter::kSuperformula
		|| opt == Parameter::kSuperformulaPoissonDisk
		|| opt == Parameter::kBccTetrahedralize ) {
		windowMenu = menuBar()->addMenu(tr("&Window"));
		windowMenu->addAction(showSFControlAct);
		
		m_superformulaControl->show();
	}
	else if(opt == Parameter::kRedblueRefine) {
		windowMenu = menuBar()->addMenu(tr("&Window"));
		windowMenu->addAction(showRBControlAct);
		
		m_redblueControl->show();
	}
}

}