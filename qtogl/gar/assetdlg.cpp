/*
 *  assetdlg.cpp
 *  garden
 *
 *  Created by jian zhang on 3/21/17.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
 
#include <QtGui>
#include <QtGui/QTreeWidget>
#include "groundAssets.h"
#include "plantAssets.h"
#include "GrassPalette.h"
#include "assetdlg.h"

using namespace std;

AssetDlg::AssetDlg(QWidget *parent)
    : QDialog(parent)
{	
	m_rgtArea = new QScrollArea(this);
	m_rgtArea->setWidgetResizable(true);
	
	m_assetTree = new QTreeWidget(this);
	m_assetTree->setMinimumHeight(120);
	m_assetTree->setAlternatingRowColors(true);
	
	m_split = new QSplitter(this);
	m_split->addWidget(m_assetTree);
	m_split->addWidget(m_rgtArea);
	
	QList<int> sizes;
	sizes<<200;
	sizes<<400;
	m_split->setSizes(sizes );
	
    QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->addWidget(m_split);

    setLayout(mainLayout);
	setWindowTitle(tr(" ") );
	connect(m_assetTree, SIGNAL(itemClicked(QTreeWidgetItem*, int)), 
			this, SLOT(onSelectAsset(QTreeWidgetItem *, int)));
	m_assetTree->setHeaderLabel(tr("Asset"));
	m_grassPlt = new GrassPalette(this);
	m_grassPlt->setVisible(false);
	lsGround();
	lsPlant();
	
}

void AssetDlg::keyPressEvent(QKeyEvent *e)
{
}

void AssetDlg::lsGround()
{
	m_groundAsset = new GroundAssets(m_assetTree);
}

void AssetDlg::lsPlant()
{
	m_plantAsset = new PlantAssets(m_assetTree);
}

void AssetDlg::onSelectAsset(QTreeWidgetItem * item, int column)
{
	QVariant varwhat = item->data(0, Qt::WhatsThisRole);
	QString swhat = varwhat.toString();
	if(swhat == tr("allGrass") ) {
		m_rgtArea->setWidget(m_grassPlt);
		m_grassPlt->setVisible(true);
	}
	if(swhat == tr("allRoot") ) {
		m_rgtArea->setWidget(m_grassPlt);
		m_grassPlt->setVisible(true);
	}
}

void AssetDlg::closeEvent ( QCloseEvent * e )
{
	emit onAssetDlgClose();
	QDialog::closeEvent(e);
}
