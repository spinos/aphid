#include <QtGui>

#include "triWindow.h"
#include "triWidget.h"

TriWindow::TriWindow(const std::string & filename)
{
    glWidget = new TriWidget(filename);
	
	setCentralWidget(glWidget);
    if(filename.size() < 1) setWindowTitle(tr("KdNTree"));
	else setWindowTitle(tr(filename.c_str()));
}
//! [1]

void TriWindow::keyPressEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Escape)
        close();

	QWidget::keyPressEvent(e);
}
