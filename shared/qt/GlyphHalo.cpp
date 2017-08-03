/*
 *  GlyphHalo.cpp
 *  
 *
 *  Created by jian zhang on 8/4/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include <QtGui>
#include "GlyphHalo.h"

namespace aphid {

GlyphHalo::GlyphHalo(QGraphicsItem * parent) : QGraphicsEllipseItem(parent)
{
	setRect(0, 0, 100, 100);
	setPen(QPen(Qt::NoPen));
	//setBrush(QColor(96, 79, 127) );
	setBrush(QColor(66, 101, 133) );
	setZValue(-1);
	//setFlag(ItemStacksBehindParent);
	hide();
}

GlyphHalo::~GlyphHalo()
{
}

}
