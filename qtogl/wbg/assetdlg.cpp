/*
 *  assetdlg.cpp
 *  wbg
 *
 *  Created by jian zhang on 3/21/17.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
 
#include <QtGui>
#include <QtGui/QTreeWidget>
#include "HeightFieldAssets.h"
#include "HeightFieldAttrib.h"
#include "assetdlg.h"

using namespace std;

AssetDlg::AssetDlg(QWidget *parent)
    : QDialog(parent)
{	
	m_split = new QSplitter(this);
	m_assetTree = new QTreeWidget(this);
	m_assetTree->setMinimumHeight(120);
	m_assetTree->setAlternatingRowColors(true);
	
	m_rgtArea = new QScrollArea(this);
	m_rgtArea->setWidgetResizable(true);
	
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
	lsHeightField();
	
	m_heightFieldAttr = new HeightFieldAttrib(this);
	m_heightFieldAttr->setVisible(false);
	
	connect(m_heightFieldAttr, SIGNAL(tranformToolChanged(int)), 
			this, SLOT(onHeightFieldTransformToolChanged(int)));
				
}

void AssetDlg::keyPressEvent(QKeyEvent *e)
{
}

void AssetDlg::lsHeightField()
{
	m_heightFieldAsset = new HeightFieldAssets(m_assetTree);
}

void AssetDlg::onSelectAsset(QTreeWidgetItem * item, int column)
{
	QVariant varwhat = item->data(0, Qt::WhatsThisRole);
	QString swhat = varwhat.toString();
	if(swhat == tr("addHeightField") ) {
		loadHeightField(item->parent() );
	}
	if(swhat == tr("fileHeightField") ) {
		selectHeightField(item);
	}
}

void AssetDlg::loadHeightField(QTreeWidgetItem * item)
{	
	QString fileName = QFileDialog::getOpenFileName(this,
			tr("Open Image"), "~", tr("EXR Image Files (*.exr)"));
			
	if(fileName.length() < 5) {
		qDebug()<<" abort loading HeightField";
		return;
	}
	
	bool stat = m_heightFieldAsset->addHeightField(fileName);
	if(stat) {
		emit onHeightFieldAdd();
	}
}

void AssetDlg::selectHeightField(QTreeWidgetItem * item)
{
	QVariant var = item->data(0, Qt::UserRole);
	int ifld = var.toInt();
	m_rgtArea->	setWidget(m_heightFieldAttr);
	m_heightFieldAttr->selHeightField(ifld);
	m_heightFieldAttr->setVisible(true);
	emit onHeightFieldSel(ifld);
}

void AssetDlg::closeEvent ( QCloseEvent * e )
{
	emit onAssetDlgClose();
	QDialog::closeEvent(e);
}

void AssetDlg::onHeightFieldTransformToolChanged(int x)
{
	emit sendHeightFieldTransformTool(x);
}
