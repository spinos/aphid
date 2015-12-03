#include <QtGui>
#include "window.h"

Window::Window(LfWorld * world)
{
   //  qDebug()<<"window";
	m_thread = new LfThread(world, this);
    m_mainWidget = new LfWidget(world, this);
	m_statistics = new StatisticDialog(world, this);
	
	setCentralWidget(m_mainWidget);
    setWindowTitle(tr("Untitled"));
	
	createActions();
    createMenus();
	
	qRegisterMetaType<QImage>("QImage");
    connect(m_thread, SIGNAL(sendInitialDictionary(QImage)),
            this, SLOT(recvInitialDictionary(QImage)));
			
	connect(m_thread, SIGNAL(sendDictionary(QImage)),
            m_mainWidget, SLOT(recvDictionary(QImage)));
			
	connect(m_thread, SIGNAL(sendSparsity(QImage)),
            m_statistics, SLOT(recvSparsity(QImage)));
			
	connect(m_thread, SIGNAL(sendPSNR(float)),
            m_statistics, SLOT(recvPSNR(float)));
			
	m_thread->initAtoms();
	m_statistics->show();
}

void Window::createMenus()
{
    /*fileMenu = menuBar()->addMenu(tr("&File"));
    fileMenu->addAction(importTriangleAct);
	fileMenu->addAction(importCurveAct);
	fileMenu->addAction(importPatchAct);*/
    windowMenu = menuBar()->addMenu(tr("&Window"));
    windowMenu->addAction(statisticAct);
}

void Window::createActions()
{
    statisticAct = new QAction(tr("&Statistics"), this);
	statisticAct->setStatusTip(tr("Sparsity and PSNR"));
    connect(statisticAct, SIGNAL(triggered()), m_statistics, SLOT(show()));
}

void Window::recvInitialDictionary(const QImage &image)
{
	m_thread->beginLearn();
}

void Window::keyPressEvent(QKeyEvent *e)
{
	// if (e->key() == Qt::Key_Escape)
       // close();

	QWidget::keyPressEvent(e);
}
