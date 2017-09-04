/*
 *   window.cpp
 *   garden
 */
#include <QtGui>

#include "window.h"
#include "widget.h"
#include "toolBox.h"
#include "ShrubScene.h"
#include "graphchart/ShrubChartView.h"
#include "ChartDlg.h"
#include "Vegetation.h"
#include "VegetationPatch.h"
#include "exportDlg.h"
#include "AboutGardenDlg.h"
#include "inout/ExportExample.h"
#include "AttribDlg.h"
#include "TexcoordDlg.h"
#include "gar_common.h"

Window::Window()
{
	m_vege = new Vegetation;
	m_vege->setSynthByAngleAlign();
	m_tools = new ToolBox(this);
	
	m_scene = new ShrubScene(m_vege, this);
	m_chartView = new ShrubChartView(m_scene);
	m_chart = new ChartDlg(m_chartView, this);
	m_attrib = new AttribDlg(m_scene, this);
	m_texcoord = new TexcoordDlg(m_scene, this);
	glWidget = new GLWidget(m_vege, m_scene, this);
	m_aboutDlg = new AboutGardenDlg(this);
	
	addToolBar(m_tools);
	
	setCentralWidget(glWidget);
    setWindowTitle(tr("Garden") );
	
	createActions();
    createMenus();
	
	connect(m_tools, SIGNAL(actionTriggered(int)), 
			this, SLOT(recvToolAction(int)));
			
	connect(m_tools, SIGNAL(dspStateChanged(int)), 
			this, SLOT(recvDspState(int)));
			
	connect(m_chart, SIGNAL(onChartDlgClose()), 
			this, SLOT(recvChartDlgClose()));
			
	connect(m_attrib, SIGNAL(onAttribDlgClose()), 
			this, SLOT(recvAttribDlgClose()));
			
	connect(m_texcoord, SIGNAL(onTexcoordDlgClose()), 
			this, SLOT(recvTexcoordDlgClose()));
			
	connect(m_scene, SIGNAL(sendSelectGlyph(bool)), 
			m_attrib, SLOT(recvSelectGlyph(bool)));
			
	connect(m_scene, SIGNAL(sendSelectGlyph(bool)), 
			m_texcoord->getWidget(), SLOT(recvSelectGlyph(bool)));
			
	connect(m_scene, SIGNAL(sendSelectGlyph(bool)), 
			glWidget, SLOT(update()));
			
	connect(m_attrib->getWidget(), SIGNAL(sendAttribChanged()), 
			glWidget, SLOT(update()));
			
	connect(m_attrib->getWidget(), SIGNAL(sendAttribChanged()), 
			m_texcoord->getWidget(), SLOT(update()));
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
	m_graphAct = new QAction(tr("&Graph"), this);
	m_graphAct->setCheckable(true);
	m_graphAct->setChecked(true);
	connect(m_graphAct, SIGNAL(toggled(bool)), this, SLOT(toggleChartDlg(bool)));
	
	m_attribAct = new QAction(tr("&Attribute"), this);
	m_attribAct->setCheckable(true);
	m_attribAct->setChecked(true);
	connect(m_attribAct, SIGNAL(toggled(bool)), this, SLOT(toggleAttribDlg(bool)));
	
	m_texcoordAct = new QAction(tr("&UV"), this);
	m_texcoordAct->setCheckable(true);
	m_texcoordAct->setChecked(false);
	connect(m_texcoordAct, SIGNAL(toggled(bool)), this, SLOT(toggleTexcoordDlg(bool)));
	
	m_exportAct = new QAction(tr("&Export"), this);
	connect(m_exportAct, SIGNAL(triggered(bool)), this, SLOT(performExport(bool)));

	m_aboutAct = new QAction(tr("&About Garden"), this);
	m_aboutAct->setMenuRole(QAction::NoRole);
	connect(m_aboutAct, SIGNAL(triggered(bool)), this, SLOT(shoAboutWin()));
}
	
void Window::createMenus()
{
	m_fileMenu = menuBar()->addMenu(tr("&File")); 
	m_fileMenu->addAction(m_exportAct);
	m_windowMenu = menuBar()->addMenu(tr("&Window")); 
	m_windowMenu->addAction(m_graphAct);
	m_windowMenu->addAction(m_attribAct);
	m_windowMenu->addAction(m_texcoordAct);
	m_helpMenu = menuBar()->addMenu(tr("&Help"));
	m_helpMenu->addAction(m_aboutAct); 
}

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

void Window::toggleTexcoordDlg(bool x)
{
	if(x) {
		m_texcoord->show();
	} else {
		m_texcoord->hide();
	}
}

void Window::recvChartDlgClose()
{ m_graphAct->setChecked(false); }

void Window::recvAttribDlgClose()
{ m_attribAct->setChecked(false); }

void Window::recvTexcoordDlgClose()
{ m_texcoordAct->setChecked(false); }

void Window::recvToolAction(int x)
{	
	switch(x) {
		case gar::actViewPlant:
			singleSynth();
		break;
		case gar::actViewTurf:
			multiSynth();
		break;
		case gar::actViewAsset:
		glWidget->setViewState(gar::actViewAsset);
		break;
	}
}

void Window::singleSynth()
{
	m_scene->genSinglePlant();
	glWidget->setViewState(gar::actViewPlant);
}

void Window::multiSynth()
{
	m_scene->genMultiPlant();
	glWidget->setViewState(gar::actViewTurf);
}

void Window::showDlgs()
{
	m_attrib->show();
	m_attrib->raise();
	m_attrib->move(0, 0);
	m_chart->show();
	m_chart->raise();
	m_chart->move(360, 300);
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

void Window::shoAboutWin()
{
	m_aboutDlg->show();
	m_aboutDlg->raise();
	m_aboutDlg->activateWindow();
}
