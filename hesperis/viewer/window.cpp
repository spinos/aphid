/*
 *   window.cpp
 *   hes viewer
 */
#include <QtGui>

#include "window.h"
#include "glWidget.h"
#include "toolBox.h"
#include <HesScene.h>

using namespace aphid;

Window::Window()
{
    m_scene = new HesScene;
	glWidget = new GLWidget(m_scene, this);
	m_tools = new ToolBox(this);
	
	addToolBar(m_tools);
	
	m_centerStack = new QStackedWidget(this);
	m_centerStack->addWidget(glWidget);
	setCentralWidget(m_centerStack);
    setWindowTitle(tr("Hesperis Viewer") );
	
	createActions();
    createMenus();
	
	connect(m_tools, SIGNAL(actionTriggered(int)), 
			this, SLOT(recvToolAction(int)));
			
	connect(m_tools, SIGNAL(dspStateChanged(int)), 
			this, SLOT(recvDspState(int)));
			
}

Window::~Window()
{}

void Window::keyPressEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Escape) {
        close();
	}
	
	QWidget::keyPressEvent(e);
}

void Window::createActions()
{
	m_assetAct = new QAction(tr("&Asset"), this);
	m_assetAct->setCheckable(true);
	m_assetAct->setChecked(true);
	connect(m_assetAct, SIGNAL(toggled(bool)), this, SLOT(toggleAssetDlg(bool)));
	
	m_loadAct = new QAction(tr("&Load"), this);
	connect(m_loadAct, SIGNAL(triggered(bool)), this, SLOT(performLoad(bool)));
}
	
void Window::createMenus()
{
	m_fileMenu = menuBar()->addMenu(tr("&File")); 
	m_fileMenu->addAction(m_loadAct);
	//m_windowMenu = menuBar()->addMenu(tr("&Window")); 
	//m_windowMenu->addAction(m_assetAct);
}

void Window::toggleAssetDlg(bool x)
{
	if(x) {
		//m_assets->show();
	} else {
		//m_assets->hide();
	}
}

void Window::recvAssetDlgClose()
{
	//m_assetAct->setChecked(false);
}

void Window::recvToolAction(int x)
{
	switch(x) {
		case 0:
		break;
		default:
		break;
	}
}

void Window::changeView(int x)
{
	if(x == m_centerStack->currentIndex() ) {
		return;
	}
	
	//if(gar::actViewPlant == x ) {
		//glWidget->setDisplayState(gar::dsTriangle);
		//m_tools->setDisplayState(gar::dsTriangle);
	//}
	
	m_centerStack->setCurrentIndex(x);
}

void Window::showAssets()
{
	//m_assets->show();
	//m_assets->raise();
	//m_assets->move(0, 0);
}

void Window::performLoad(bool x)
{
    QString fileName = QFileDialog::getOpenFileName(this,
			tr("Select one file to load"), 
			"~", 
			tr("OFL Cache Files (*.m *.hes *.h5)"));
	
	if(fileName.length() < 5) {
	    qDebug()<<" abort loading file ";
		return;
	}
	
	qDebug()<<fileName;
	bool stat = m_scene->load(fileName.toStdString() );
	if(stat) {
	    qDebug()<<" hes is loaded ";
	}
}

void Window::recvDspState(int x)
{
	//if(gar::actViewPlant != m_centerStack->currentIndex() ) {
	//	return;
	//}
	
	glWidget->setDisplayState(x);
}
