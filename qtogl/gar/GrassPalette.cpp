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
#include "AssetDescription.h"
#include "PiecesList.h"
#include <qt/ContextIconFrame.h>
#include <gar_common.h>
#include <boost/format.hpp>

using namespace aphid;

GrassPalette::GrassPalette(QWidget *parent) : QWidget(parent)
{
	m_grassList = new PiecesList(this);
	m_describ = new AssetDescription(this);
	
	QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->addWidget(m_grassList);
	mainLayout->addWidget(m_describ);
	mainLayout->setContentsMargins(0,0,0,0);
	setLayout(mainLayout);
	
	connect(m_grassList, SIGNAL(itemClicked(QListWidgetItem *) ), 
		this, SLOT(selectAGrass(QListWidgetItem *) ) );
		
	connect(this, SIGNAL(onGrassSel(int) ), 
		m_describ, SLOT(recvAssetSel(int) ) );

}

void GrassPalette::selectAGrass(QListWidgetItem * item)
{
	QVariant di = item->data(Qt::UserRole+1);
	emit onGrassSel(di.toInt());
}
