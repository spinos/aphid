#include <QtGui>

#include "glwidget.h"
#include "window.h"

//! [0]
Window::Window(const std::string & filename)
{
    glWidget = new GLWidget(filename);
	
	setCentralWidget(glWidget);
    if(filename.size() < 1) setWindowTitle(tr("KdNTree"));
	else setWindowTitle(tr(filename.c_str()));
}
//! [1]

void Window::keyPressEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Escape)
        close();

	QWidget::keyPressEvent(e);
}
