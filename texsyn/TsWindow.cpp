#include <QtGui>
#include "TsWindow.h"

namespace tss {

TsWindow::TsWindow(LfMachine * world)
{
	m_thread = new LfThread(world, this);
    m_statistics = new StatisticDialog(world, this);
	m_dictionary = new DictionaryDialog(world, this);
	m_image = new ImageDialog(world, this);
	
	setWindowTitle(tr("Untitled"));
	
	createActions();
    createMenus();
	
	if(!world->param()->isValid() ) 
		return;
	
	qRegisterMetaType<QImage>("QImage");
    connect(m_thread, SIGNAL(sendInitialDictionary(QImage)),
            this, SLOT(recvInitialDictionary(QImage)));
			
	connect(m_thread, SIGNAL(sendDictionary(QImage)),
            m_dictionary, SLOT(recvDictionary(QImage)));
			
	connect(m_thread, SIGNAL(sendPSNR(float)),
            m_statistics, SLOT(recvPSNR(float)));
	
	connect(m_thread, SIGNAL(sendIterDone(int)),
            m_statistics, SLOT(recvIterDone(int)));
			
	m_thread->initAtoms();
	m_statistics->show();
}

TsWindow::~TsWindow()
{
	qDebug()<<"closing main window";
	delete m_thread;
	delete m_statistics;
	delete m_dictionary;
	delete m_image;
}

void TsWindow::createMenus()
{
	generateMenu = menuBar()->addMenu(tr("&Generate"));
	generateMenu->addAction(generateAct);
    windowMenu = menuBar()->addMenu(tr("&Window"));
	windowMenu->addAction(shoImageAct);
    windowMenu->addAction(shoDictionaryAct);
	windowMenu->addAction(statisticAct);
}

void TsWindow::createActions()
{
	generateAct = new QAction(tr("&Stop"), this);
	generateAct->setStatusTip(tr("Stop learning to generate"));
	connect(generateAct, SIGNAL(triggered()), m_thread, SLOT(endLearn()));
	
	shoImageAct = new QAction(tr("&Image"), this);
	shoImageAct->setStatusTip(tr("Show Image"));
	connect(shoImageAct, SIGNAL(triggered()), m_image, SLOT(show()));
	
	shoDictionaryAct = new QAction(tr("&Dictionary"), this);
	shoDictionaryAct->setStatusTip(tr("Show Dictionary"));
	connect(shoDictionaryAct, SIGNAL(triggered()), m_dictionary, SLOT(show()));
	
    statisticAct = new QAction(tr("&Statistics"), this);
	statisticAct->setStatusTip(tr("Show PSNR"));
    connect(statisticAct, SIGNAL(triggered()), m_statistics, SLOT(show()));
}

void TsWindow::recvInitialDictionary(const QImage &image)
{
	m_thread->beginLearn();
}

void TsWindow::keyPressEvent(QKeyEvent *e)
{
	// if (e->key() == Qt::Key_Escape)
       // close();

	QWidget::keyPressEvent(e);
}

}
