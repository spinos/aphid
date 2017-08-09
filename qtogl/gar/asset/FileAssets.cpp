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
	
	setExpanded(true);

}