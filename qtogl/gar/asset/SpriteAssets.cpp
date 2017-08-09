/*
 *  SpriteAssets.cpp
 *  garden
 *
 *  Created by jian zhang on 3/22/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include <QtGui>
#include "SpriteAssets.h"

//using namespace aphid;

SpriteAssets::SpriteAssets(QTreeWidget *parent) : QTreeWidgetItem(parent)
{
	QIcon billboardIcon(":/icons/texturesprite.png");
	setText(0, tr("Billboard"));
	setIcon(0, billboardIcon);
	setData(0, Qt::WhatsThisRole, QString(tr("allSprite")) );
	
	setExpanded(true);

}