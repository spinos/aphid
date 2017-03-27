/*
 *  GrassPalette.cpp
 *  
 *
 *  Created by jian zhang on 3/22/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include <QtGui>
#include "GrassPalette.h"
#include <qt/ContextIconFrame.h>
#include <gar_common.h>
#include <boost/format.hpp>

using namespace aphid;

GrassPalette::GrassPalette(QWidget *parent) : QWidget(parent)
{
	m_grassList = new QListWidget(this);
	m_grassList->setViewMode(QListView::IconMode);
    m_grassList->setIconSize(QSize(32, 32));
	m_grassList->setSpacing(4);
	
	QListWidgetItem *pieceItem = new QListWidgetItem(m_grassList);
	QIcon cloverIcon(":/icons/clover.png");
	pieceItem->setIcon(cloverIcon);
	pieceItem->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable);
	
	QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->addWidget(m_grassList);
	mainLayout->setContentsMargins(0,0,0,0);
	setLayout(mainLayout);
	
	connect(m_grassList, SIGNAL(itemClicked(QListWidgetItem *) ), 
		this, SLOT(selectAGrass(QListWidgetItem *) ) );

}

void GrassPalette::selectAGrass(QListWidgetItem * item)
{
	qDebug()<<"GrassPalette::selectAGrass"<<item;
	
}
