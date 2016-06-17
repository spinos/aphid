#include <QtGui>

#include "glwidget.h"
#include "window.h"
#include "Delaunay2D.h"
#include "Delaunay3D.h"
#include "Hilbert2D.h"
#include "Hilbert3D.h"
#include "Bcc3dTest.h"
#include "SuperformulaTest.h"

using namespace ttg;

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
	else 
		sc = new SuperformulaTest;
		
    glWidget = new GLWidget(sc, this);
	
	setCentralWidget(glWidget);
    setWindowTitle(tr(sc->titleStr() ) );
}
//! [1]

void Window::keyPressEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Escape)
        close();

	QWidget::keyPressEvent(e);
}
