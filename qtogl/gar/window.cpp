/*
 *   window.cpp
 *   garden
 */
#include <QtGui>

#include "window.h"
#include "widget.h"
#include "toolBox.h"
#include "assetdlg.h"
#include "ShrubScene.h"
#include "ShrubChartView.h"
#include "Vegetation.h"
#include "VegetationPatch.h"
#include "exportDlg.h"
#include "ExportExample.h"
#include "gar_common.h"

Window::Window()
{
	m_vege = new Vegetation;
	glWidget = new GLWidget(m_vege, this);
	m_tools = new ToolBox(this);
	m_assets = new AssetDlg(this);
	
	m_scene = new ShrubScene(m_vege, this);
	m_chartView = new ShrubChartView(m_scene);
	
	addToolBar(m_tools);
	
	m_centerStack = new QStackedWidget(this);
	m_centerStack->addWidget(m_chartView);
	m_centerStack->addWidget(glWidget);
	setCentralWidget(m_centerStack);
    setWindowTitle(tr("Garden") );
	
	createActions();
    createMenus();
	
	connect(m_tools, SIGNAL(actionTriggered(int)), 
			this, SLOT(recvToolAction(int)));
			
	connect(m_tools, SIGNAL(dspStateChanged(int)), 
			this, SLOT(recvDspState(int)));
			
	connect(m_assets, SIGNAL(onAssetDlgClose()), 
			this, SLOT(recvAssetDlgClose()));
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
	
	m_exportAct = new QAction(tr("&Export"), this);
	connect(m_exportAct, SIGNAL(triggered(bool)), this, SLOT(performExport(bool)));
}
	
void Window::createMenus()
{
	m_fileMenu = menuBar()->addMenu(tr("&File")); 
	m_fileMenu->addAction(m_exportAct);
	m_windowMenu = menuBar()->addMenu(tr("&Window")); 
	m_windowMenu->addAction(m_assetAct);
}

void Window::toggleAssetDlg(bool x)
{
	if(x) {
		m_assets->show();
	} else {
		m_assets->hide();
	}
}

void Window::recvAssetDlgClose()
{
	m_assetAct->setChecked(false);
}

void Window::recvToolAction(int x)
{
	switch(x) {
		case gar::actViewPlant:
			singleSynth();
			changeView(x);
		break;
		case gar::actViewTurf:
			multiSynth();
			changeView(gar::actViewPlant);
		break;
		case gar::actViewGraph:
			changeView(x);
		break;
	}
}

void Window::singleSynth()
{
	m_scene->genSinglePlant();
	glWidget->update();
}

void Window::multiSynth()
{
	m_scene->genMultiPlant();
	glWidget->update();
}

void Window::changeView(int x)
{
	if(x == m_centerStack->currentIndex() ) {
		return;
	}
	
	if(gar::actViewPlant == x ) {
		glWidget->setDisplayState(gar::dsTriangle);
		m_tools->setDisplayState(gar::dsTriangle);
	}
	
	m_centerStack->setCurrentIndex(x);
}

void Window::showAssets()
{
	m_assets->show();
	m_assets->raise();
	m_assets->move(0, 0);
}

void Window::performExport(bool x)
{
	ExportDlg dlg(m_vege, this);
	int res = dlg.exec();
	if(res == QDialog::Rejected) {
		qDebug()<<"abort export";
		return;
	}
	
	ExportExample xpt(m_vege);
	xpt.exportToFile(dlg.exportToFilename());
}

void Window::recvDspState(int x)
{
	if(gar::actViewPlant != m_centerStack->currentIndex() ) {
		return;
	}
	
	glWidget->setDisplayState(x);
}
