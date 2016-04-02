#include <QtGui>

#include "window.h"
#include "glwidget.h"
#include "wldWidget.h"
#include "triWidget.h"

Window::Window(int argc, char *argv[])
{
	if(argc < 3) {
		glWidget = new GLWidget("");
		setWindowTitle(tr("KdNTree"));
	}
	else {
		std::string filename(argv[argc - 1]);
		if(strcmp(argv[1], "-a") == 0)
			glWidget = new TriWidget(filename);
		else
			glWidget = new WldWidget(filename);
		setWindowTitle(tr(filename.c_str()));
	}
	
	setCentralWidget(glWidget);
}
//! [1]

void Window::keyPressEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Escape)
        close();

	QWidget::keyPressEvent(e);
}
