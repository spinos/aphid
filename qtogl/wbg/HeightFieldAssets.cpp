/*
 *  HeightFieldAssets.cpp
 *  
 *
 *  Created by jian zhang on 3/22/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include <QtGui>
#include "HeightFieldAssets.h"
#include <ttg/GlobalElevation.h>

using namespace aphid;

HeightFieldAssets::HeightFieldAssets(QTreeWidget *parent) : QTreeWidgetItem(parent)
{
	QIcon fldIcon(":/icons/heightField.png");
	setText(0, tr("Height Field"));
	setIcon(0, fldIcon);
	setData(0, Qt::WhatsThisRole, QString(tr("allHeightField")) );
	
	QIcon addIcon(":/icons/generic_add.png");
	QTreeWidgetItem *addfld = new QTreeWidgetItem(this);
	addfld->setText(0, tr("Add"));
	addfld->setIcon(0, addIcon);
	addfld->setData(0, Qt::WhatsThisRole, QString(tr("addHeightField")) );
	
	setExpanded(true);
}

bool HeightFieldAssets::addHeightField(const QString & fileName)
{
	bool stat = ttg::GlobalElevation::LoadHeightField(fileName.toStdString() );
	if(!stat) {
		qDebug()<<" not a HeightField";
		return false;
	}
	
	const std::string sfn = ttg::GlobalElevation::LastFileBaseName();
	QTreeWidgetItem *ffld = new QTreeWidgetItem;
	ffld->setText(0, tr(sfn.c_str()));
	ffld->setData(0, Qt::WhatsThisRole, QString(tr("fileHeightField")) );
	ffld->setData(0, Qt::UserRole, QVariant(ttg::GlobalElevation::NumHeightFields() - 1) );
	
	insertChild(childCount()-1, ffld);
	return true;
}