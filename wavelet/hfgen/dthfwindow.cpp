/*
 *   dthfwindow.cpp
 *
 */
 
#include <QtGui>

#include "dthfwindow.h"
#include "dthfwidget.h"
#include "SynthControl.h"
#include <img/ExrImage.h>

using namespace aphid;

DthfWindow::DthfWindow(int argc, char *argv[])
{
	ExrImage img;
	if(argc < 2) {
		std::cout<<"\n hfgen requires input filename.";
	} else {
		img.read(argv[1]);
		img.verbose();
	}
	
    m_plot = new DthfWidget(&img, this);
	m_control = new SynthControl(&img, this);
	
	setCentralWidget(m_plot);
    setWindowTitle(tr("Dual Tree Discrete Wavelet Demo"));
	
	m_control->show();
	connect(m_control, SIGNAL(l0ScaleChanged(double)), m_plot, SLOT(recvL0scale(double)));
    connect(m_control, SIGNAL(l1ScaleChanged(double)), m_plot, SLOT(recvL1scale(double)));
    connect(m_control, SIGNAL(l2ScaleChanged(double)), m_plot, SLOT(recvL2scale(double)));
    connect(m_control, SIGNAL(l3ScaleChanged(double)), m_plot, SLOT(recvL3scale(double)));
    
}

void DthfWindow::keyPressEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Escape) {
        close();
	}

	QWidget::keyPressEvent(e);
}
