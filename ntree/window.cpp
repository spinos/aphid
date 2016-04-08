#include <QtGui>

#include "window.h"
#include "glwidget.h"
#include "wldWidget.h"
#include "triWidget.h"
#include "voxWidget.h"

Window::Window(int argc, char *argv[])
{
	if(argc < 2) {
		printHelp();
		return;
	}
	else if(argc < 3) {
		if(strcmp(argv[1], "-h") == 0) {
			printHelp();
			return;
		}
		else if(strcmp(argv[1], "-tv") == 0) {
			glWidget = new VoxWidget();
			setWindowTitle(tr("Sub Voxel Test"));
		}
		else {
			printHelp();
			return;
		}
	}
	else {
		std::string filename(argv[argc - 1]);
		if(strcmp(argv[1], "-a") == 0)
			glWidget = new TriWidget(filename);
		else if(strcmp(argv[1], "-w") == 0)
			glWidget = new WldWidget(filename);
		else {
			printHelp();
			return;
		}
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

void Window::printHelp()
{
	std::cout<<"\n testntree [options] filename"
			<<"\n -h print this information"
			<<"\n -a view asset"
			<<"\n -w view world"
			<<"\n -tv test voxel"
			<<"\n end \n";
	std::cout.flush();
}
