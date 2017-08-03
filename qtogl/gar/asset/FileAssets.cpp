/*
 *  FileAssets.cpp
 *  garden
 *
 *  Created by jian zhang on 3/22/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include <QtGui>
#include "FileAssets.h"

//using namespace aphid;

FileAssets::FileAssets(QTreeWidget *parent) : QTreeWidgetItem(parent)
{
	QIcon fileIcon(":/icons/file.png");
	setText(0, tr("Files"));
	setIcon(0, fileIcon);
	setData(0, Qt::WhatsThisRole, QString(tr("allFile")) );
	
	//QIcon grassIcon(":/icons/ground.png");
	//QTreeWidgetItem * grass = new QTreeWidgetItem(this);
	//grass->setText(0, tr("Ground"));
	//grass->setIcon(0, grassIcon);
	//grass->setData(0, Qt::WhatsThisRole, QString(tr("allGround")) );
	
	setExpanded(true);

}