#include <QtGui>
#include "window.h"

Window::Window(LfMachine * world)
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
	
	connect(m_thread, SIGNAL(sendIterDone(int)),
            m_statistics, SLOT(recvIterDone(int)));
			
	m_thread->initAtoms();
	m_statistics->show();
}

Window::~Window()
{
	delete m_thread;
	delete m_statistics;
}

void Window::createMenus()
{
    /*fileMenu = menuBar()->addMenu(tr("&File"));
    fileMenu->addAction(importTriangleAct);
	fileMenu->addAction(importCurveAct);
	fileMenu->addAction(importPatchAct);*/
	generateMenu = menuBar()->addMenu(tr("&Generate"));
	generateMenu->addAction(generateAct);
    windowMenu = menuBar()->addMenu(tr("&Window"));
    windowMenu->addAction(statisticAct);
}

void Window::createActions()
{
	generateAct = new QAction(tr("&Stop"), this);
	generateAct->setStatusTip(tr("Stop learning to generate"));
	connect(generateAct, SIGNAL(triggered()), m_thread, SLOT(endLearn()));
    statisticAct = new QAction(tr("&Statistics"), this);
	statisticAct->setStatusTip(tr("PSNR"));
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
