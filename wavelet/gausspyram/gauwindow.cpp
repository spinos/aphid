/*
 *   gauwindow.cpp
 *
 */
 
#include <QtGui>

#include "gauwindow.h"
#include "gauwidget.h"
#include <img/ExrImage.h>

using namespace aphid;

GauWindow::GauWindow(int argc, char *argv[])
{
	ExrImage img;
	if(argc < 2) {
		std::cout<<"\n hfgen requires input filename.";
	} else {
		img.read(argv[1]);
		img.verbose();
	}
	
    m_plot = new GauWidget(&img, this);
	
	setCentralWidget(m_plot);
    setWindowTitle(tr("Gaussian Pyramid Demo"));
	
}

void GauWindow::keyPressEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Escape) {
        close();
	}

	QWidget::keyPressEvent(e);
}
