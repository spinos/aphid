/*
 *  GardenGlyph.cpp
 *  
 *
 *  Created by jian zhang on 3/31/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include <QtGui>
#include "GardenGlyph.h"
#include "GlyphPort.h"

GardenGlyph::GardenGlyph(const QPixmap & iconPix,
			QGraphicsItem * parent) : QGraphicsPathItem(parent)
{
	resizeBlock(120, 36);
	setPen(QPen(Qt::darkGray));
	setBrush(Qt::lightGray);
	setFlag(QGraphicsItem::ItemIsMovable);
	setFlag(QGraphicsItem::ItemIsSelectable);
	
	m_icon = new QGraphicsPixmapItem(iconPix, this);
	m_icon->setPos(60-16, 18-16);
	
}

void GardenGlyph::resizeBlock(int bx, int by)
{
	QPainterPath p;
	p.addRoundedRect(0, 0, bx, by, 4, 4);
	setPath(p);
	m_blockWidth = bx;
	m_blockHeight = by;
}

void GardenGlyph::centerIcon()
{
	m_icon->setPos(m_blockWidth/2 - 16, m_blockHeight/2 - 16);
}

void GardenGlyph::movePorts(int n, bool downSide)
{
	if(n < 1) {
		return;
	}
	
	int py = -6;
	if(downSide) {
		py = m_blockHeight + 6;
	}
	
	int px = m_blockWidth / 2 - 30 * (n / 2);
	if((n & 1) == 0) {
		px += 15;
	}
	
	foreach(QGraphicsItem *port_, children()) {
		if (port_->type() != GlyphPort::Type) {
			continue;
		}

		GlyphPort *port = (GlyphPort*) port_;
		
		if(port->isOutgoing() == downSide ) {
			
			port->setPos(px, py);
			px += 30;
		}
	}
}	

GlyphPort * GardenGlyph::addPort(const QString & name, 
							bool isOutgoing)
{
	GlyphPort * pt = new GlyphPort(this);
	pt->setPortName(name);
	pt->setIsOutgoing(isOutgoing);
	return pt;
}

void GardenGlyph::finalizeShape()
{
	int nincoming = 0;
	int noutgoing = 0;
	foreach(QGraphicsItem *port_, children()) {
		if (port_->type() != GlyphPort::Type) {
			continue;
		}

		GlyphPort *port = (GlyphPort*) port_;
		
		if(port->isOutgoing() ) {
			noutgoing++;
		} else {
			nincoming++;
		}
	}

	int wup = nincoming * 30;
	int wdown = noutgoing * 30;
	if(wup > wdown) {
		wup = wdown;
	}
	if(wup > 120) {
		resizeBlock(wup, 36);
		centerIcon();
	}
	
	movePorts(nincoming, false);
	movePorts(noutgoing, true);
	
}

void GardenGlyph::moveBlockBy(const QPointF & dp)
{
	foreach(QGraphicsItem *port_, children()) {
		if (port_->type() != GlyphPort::Type) {
			continue;
		}

		GlyphPort *port = (GlyphPort*) port_;
		port->updateConnectionsPath();
		
	}
	moveBy(dp.x(), dp.y() );
}
