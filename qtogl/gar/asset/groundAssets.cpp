/*
 *  groundAssets.cpp
 *  garden
 *
 *  Created by jian zhang on 3/22/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include <QtGui>
#include "groundAssets.h"

//using namespace aphid;

GroundAssets::GroundAssets(QTreeWidget *parent) : QTreeWidgetItem(parent)
{
	QIcon plantIcon(":/icons/spade.png");
	setText(0, tr("Utilities"));
	setIcon(0, plantIcon);
	setData(0, Qt::WhatsThisRole, QString(tr("allUtility")) );
	
	QIcon grassIcon(":/icons/ground.png");
	QTreeWidgetItem * grass = new QTreeWidgetItem(this);
	grass->setText(0, tr("Ground"));
	grass->setIcon(0, grassIcon);
	grass->setData(0, Qt::WhatsThisRole, QString(tr("allGround")) );
	
	setExpanded(true);

}