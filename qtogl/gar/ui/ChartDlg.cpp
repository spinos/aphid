/*
 *  ChartDlg.cpp
 *  garden
 *
 *  Created by jian zhang on 8/4/17.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
 
#include <QtGui>
#include "ChartDlg.h"
#include "PaletteView.h"
#include <graphchart/ShrubChartView.h>
#include <asset/groundAssets.h>
#include <asset/plantAssets.h>
#include <asset/FileAssets.h>
#include <asset/SpriteAssets.h>

using namespace std;

ChartDlg::ChartDlg(ShrubChartView* chart, QWidget *parent)
    : QDialog(parent)
{	
	m_chartView = new PaletteView(chart, this);
	
	m_assetTree = new QTreeWidget(this);
	m_assetTree->setMinimumHeight(120);
	m_assetTree->setAlternatingRowColors(true);
	m_assetTree->setHeaderLabel(tr("Asset"));
	lsAssets();
	
	m_split = new QSplitter(this);
	m_split->addWidget(m_assetTree);
	m_split->addWidget(m_chartView);
	
	QList<int> sizes;
	sizes<<120;
	sizes<<540;
	m_split->setSizes(sizes );
	
    QVBoxLayout *mainLayout = new QVBoxLayout;
	mainLayout->setContentsMargins(0,0,0,0);
    mainLayout->addWidget(m_split);

    setLayout(mainLayout);
	setWindowTitle(tr("Graph View") );
	
	connect(m_assetTree, SIGNAL(itemClicked(QTreeWidgetItem*, int)), 
			this, SLOT(onSelectAsset(QTreeWidgetItem *, int)));
}

void ChartDlg::keyPressEvent(QKeyEvent *e)
{}

void ChartDlg::closeEvent ( QCloseEvent * e )
{
	emit onChartDlgClose();
	QDialog::closeEvent(e);
}

void ChartDlg::lsAssets()
{
	new GroundAssets(m_assetTree);
	new PlantAssets(m_assetTree);
	new SpriteAssets(m_assetTree);
	new FileAssets(m_assetTree);
}

void ChartDlg::onSelectAsset(QTreeWidgetItem * item, int column)
{
	QVariant varwhat = item->data(0, Qt::WhatsThisRole);
	QString swhat = varwhat.toString();
	m_chartView->showNamedPieces(swhat);
}
