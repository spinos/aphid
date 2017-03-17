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
#include <QTreeWidgetItem>
#include "assetdlg.h"

using namespace std;

AssetDlg::AssetDlg(QWidget *parent)
    : QDialog(parent)
{	
	m_assetTree = new QTreeWidget(this);
	m_assetTree->setMinimumHeight(120);
	m_assetTree->setAlternatingRowColors(true);
	
    QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->addWidget(m_assetTree);

    setLayout(mainLayout);
	setWindowTitle(tr("Asset") );
	connect(m_assetTree, SIGNAL(itemClicked(QTreeWidgetItem*, int)), 
			this, SLOT(onSelectAHistoryTexture(QTreeWidgetItem *, int)));
	
}

void AssetDlg::onSelectATexture(QString texname)
{
}

void AssetDlg::keyPressEvent(QKeyEvent *e)
{
}