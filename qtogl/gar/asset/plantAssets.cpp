/*
 *  PlantAssets.cpp
 *  garden
 *
 *  Created by jian zhang on 3/22/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include <QtGui>
#include "plantAssets.h"

//using namespace aphid;

PlantAssets::PlantAssets(QTreeWidget *parent) : QTreeWidgetItem(parent)
{
	QIcon plantIcon(":/icons/plant.png");
	setText(0, tr("Plant"));
	setIcon(0, plantIcon);
	setData(0, Qt::WhatsThisRole, QString(tr("allPlant")) );
	
	QIcon grassIcon(":/icons/grass.png");
	QTreeWidgetItem * grass = new QTreeWidgetItem(this);
	grass->setText(0, tr("Grass"));
	grass->setIcon(0, grassIcon);
	grass->setData(0, Qt::WhatsThisRole, QString(tr("allGrass")) );
	
	QIcon stemIcon(":/icons/stem.png");
	QTreeWidgetItem * stem = new QTreeWidgetItem(this);
	stem->setText(0, tr("Stem"));
	stem->setIcon(0, stemIcon);
	stem->setData(0, Qt::WhatsThisRole, QString(tr("allStem")) );
	
	QIcon twigIcon(":/icons/twig.png");
	QTreeWidgetItem * twig = new QTreeWidgetItem(this);
	twig->setText(0, tr("Twig"));
	twig->setIcon(0, twigIcon);
	twig->setData(0, Qt::WhatsThisRole, QString(tr("allTwig")) );
	
	setExpanded(true);

}