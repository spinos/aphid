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
#include "gar_common.h"

Window::Window()
{
	m_vege = new Vegetation;
	glWidget = new GLWidget(m_vege, this);
	m_tools = new ToolBox(this);
	m_assets = new AssetDlg(this);
	
	m_scene = new ShrubScene(this);
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
	
}
	
void Window::createMenus()
{
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
	m_scene->genPlants(m_vege->patch(0));
	m_vege->setNumPatches(1);
	glWidget->update();
}

void Window::multiSynth()
{
	const int n = m_vege->getMaxNumPatches();
	for(int i=0;i<n;++i) {
		m_scene->genPlants(m_vege->patch(i));
	}
	m_vege->setNumPatches(n);
	m_vege->rearrange();
	glWidget->update();
}

void Window::changeView(int x)
{
	if(x == m_centerStack->currentIndex() ) {
		return;
	}
	
	m_centerStack->setCurrentIndex(x);
}

void Window::showAssets()
{
	m_assets->show();
	m_assets->raise();
	m_assets->move(0, 0);
}
