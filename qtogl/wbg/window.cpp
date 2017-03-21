/*
 *   window.cpp
 *   world block grid
 */
#include <QtGui>

#include "window.h"
#include "widget.h"
#include "toolBox.h"
#include "assetdlg.h"

Window::Window()
{
	glWidget = new GLWidget(this);
	m_tools = new ToolBox(this);
	m_assets = new AssetDlg(this);
	
	addToolBar(m_tools);
	setCentralWidget(glWidget);
    setWindowTitle(tr("world block grid") );
	
	createActions();
    createMenus();
	
	connect(m_tools, SIGNAL(actionTriggered(int)), 
			glWidget, SLOT(recvToolAction(int)));
			
	connect(m_assets, SIGNAL(onAssetDlgClose()), 
			this, SLOT(recvAssetDlgClose()));
			
	connect(m_assets, SIGNAL(onHeightFieldAdd()), 
			glWidget, SLOT(recvHeightFieldAdd()));
			
	connect(m_assets, SIGNAL(onHeightFieldSel(int)), 
			glWidget, SLOT(recvHeightFieldSel(int)));
			
	connect(m_assets, SIGNAL(sendHeightFieldTransformTool(int)), 
			glWidget, SLOT(recvHeightFieldTransformTool(int)));
			
}

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
	m_assetAct->setChecked(false);
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
