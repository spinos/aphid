/*
 *  GardenConnection.h
 *
 *  access to node attrib
 *
 *  Created by jian zhang on 4/1/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */
 
#include <QtGui>
#include "GardenConnection.h"
#include "GardenGlyph.h"
#include <qt/GlyphPort.h>
#include <attr/PieceAttrib.h>
#include "gar_common.h"

using namespace aphid;
   
GardenConnection::GardenConnection(QGraphicsItem * parent) : GlyphConnection(parent)
{}

GardenConnection::~GardenConnection()
{}

bool GardenConnection::canConnectTo(GlyphPort* p1) const
{ 
    QGraphicsItem * item0 = port0()->parentItem();
    QGraphicsItem * item1 = p1->parentItem();
    GardenGlyph* node0 = static_cast<GardenGlyph*>(item0);
    GardenGlyph* node1 = static_cast<GardenGlyph*>(item1);
    
	const PieceAttrib* attr0 = node0->attrib();
    const PieceAttrib* attr1 = node1->attrib();
	return attr1->canConnectToViaPort(attr0, p1->portName().toStdString() );
}

GardenGlyph* GardenConnection::node0() const
{ return PortToNode(port0()); }

GardenGlyph* GardenConnection::node1() const
{ return PortToNode(port1()); }

GardenGlyph * GardenConnection::PortToNode(const GlyphPort * pt)
{ return static_cast<GardenGlyph *>(pt->parentItem() ); }

