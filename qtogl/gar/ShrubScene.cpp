/*
 *  ShrubScene.cpp
 *  garden
 *
 *  Created by jian zhang on 3/30/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include <QtGui>
#include "ShrubScene.h"
#include "GardenGlyph.h"
#include "gar_common.h"

using namespace gar;

ShrubScene::ShrubScene(QObject *parent)
    : QGraphicsScene(parent)
{
}

void ShrubScene::genPlants()
{
	foreach(QGraphicsItem *its_, items()) {
		
		if(its_->type() == GardenGlyph::Type) {
			GardenGlyph *g = (GardenGlyph*) its_;
		
			if(g->glyphType() == gtPot) {
				qDebug()<<" pot "<<g->glyphType();
			}
		}
	}
}