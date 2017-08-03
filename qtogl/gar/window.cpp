/*
 *   window.cpp
 *   garden
 */
#include <QtGui>

#include "window.h"
#include "widget.h"
#include "toolBox.h"
#include "asset/assetdlg.h"
#include "ShrubScene.h"
#include "graphchart/ShrubChartView.h"
#include "graphchart/ChartDlg.h"
#include "Vegetation.h"
#include "VegetationPatch.h"
#include "inout/exportDlg.h"
#include "inout/ExportExample.h"
#include "attr/AttribDlg.h"
#include "gar_common.h"

Window::Window()
{
	m_vege = new Vegetation;
	m_vege->setSynthByAngleAlign();
	glWidget = new GLWidget(m_vege, this);
	m_tools = new ToolBox(this);
	m_assets = new AssetDlg(this);
	
	m_scene = new ShrubScene(m_vege, this);
	m_chartView = new ShrubChartView(m_scene);
	m_chart = new ChartDlg(m_chartView, this);
	m_attrib = new AttribDlg(m_scene, this);
	
	addToolBar(m_tools);
	
	setCentralWidget(glWidget);
    setWindowTitle(tr("Garden") );
	
	createActions();
    createMenus();
	
	connect(m_tools, SIGNAL(actionTriggered(int)), 
			this, SLOT(recvToolAction(int)));
			
	connect(m_tools, SIGNAL(dspStateChanged(int)), 
			this, SLOT(recvDspState(int)));
			
	connect(m_assets, SIGNAL(onAssetDlgClose()), 
			this, SLOT(recvAssetDlgClose()));
			
	connect(m_chart, SIGNAL(onChartDlgClose()), 
			this, SLOT(recvChartDlgClose()));
			
	connect(m_attrib, SIGNAL(onAttribDlgClose()), 
			this, SLOT(recvAttribDlgClose()));
			
	connect(m_chartView, SIGNAL(sendSelectGlyph(bool)), 
			m_attrib, SLOT(recvSelectGlyph(bool)));
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
	
	m_graphAct = new QAction(tr("&Graph"), this);
	m_graphAct->setCheckable(true);
	m_graphAct->setChecked(true);
	connect(m_graphAct, SIGNAL(toggled(bool)), this, SLOT(toggleChartDlg(bool)));
	
	m_attribAct = new QAction(tr("&Attribute"), this);
	m_attribAct->setCheckable(true);
	m_attribAct->setChecked(false);
	connect(m_attribAct, SIGNAL(toggled(bool)), this, SLOT(toggleAttribDlg(bool)));
	
	m_exportAct = new QAction(tr("&Export"), this);
	connect(m_exportAct, SIGNAL(triggered(bool)), this, SLOT(performExport(bool)));
}
	
void Window::createMenus()
{
	m_fileMenu = menuBar()->addMenu(tr("&File")); 
	m_fileMenu->addAction(m_exportAct);
	m_windowMenu = menuBar()->addMenu(tr("&Window")); 
	m_windowMenu->addAction(m_graphAct);
	m_windowMenu->addAction(m_assetAct);
	m_windowMenu->addAction(m_attribAct);
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
{ m_assetAct->setChecked(false); }

void Window::toggleChartDlg(bool x)
{
	if(x) {
		m_chart->show();
	} else {
		m_chart->hide();
	}
}

void Window::toggleAttribDlg(bool x)
{
	if(x) {
		m_attrib->show();
	} else {
		m_attrib->hide();
	}
}

void Window::recvChartDlgClose()
{ m_graphAct->setChecked(false); }

void Window::recvAttribDlgClose()
{ m_attribAct->setChecked(false); }

void Window::recvToolAction(int x)
{
	switch(x) {
		case gar::actViewPlant:
			singleSynth();
		break;
		case gar::actViewTurf:
			multiSynth();
		break;
		case gar::actViewGraph:
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

void Window::showDlgs()
{
	m_assets->show();
	m_assets->raise();
	m_assets->move(0, 0);
	m_chart->show();
	m_chart->raise();
	m_chart->move(400, 300);
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
{ glWidget->setDisplayState(x); }
