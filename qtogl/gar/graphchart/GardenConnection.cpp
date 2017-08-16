/*
 *  GardenConnection.h
 *
 *  with can make connection check
 *
 *  Created by jian zhang on 4/1/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */
 
#include <QtGui>
#include "GardenConnection.h"
#include "GardenGlyph.h"
#include <qt/GlyphPort.h>
#include "gar_common.h"

using namespace aphid;
   
GardenConnection::GardenConnection(QGraphicsItem * parent) : GlyphConnection(parent)
{
}

GardenConnection::~GardenConnection()
{}

bool GardenConnection::canConnectTo(GlyphPort* p1) const
{ 
    QGraphicsItem * item0 = port0()->parentItem();
    QGraphicsItem * item1 = p1->parentItem();
    GardenGlyph* node0 = static_cast<GardenGlyph*>(item0);
    GardenGlyph* node1 = static_cast<GardenGlyph*>(item1);
    
    if(rejectedByNode(node0, node1) )    
        return false;
    
    return true; 
}

static const int sVariableNodeTypes[3] = {gar::gtSplineSprite,
gar::gtRibSprite,
gar::gtSplineCylinder,
};

bool GardenConnection::isNodeVariable(int gt) const
{
	for(int i=0;i<3;++i) {
		if(gt == sVariableNodeTypes[i])
			return true;
	}
	return false;
}

bool GardenConnection::rejectedByNode(GardenGlyph* node0, GardenGlyph* node1) const
{
    if(gar::ToGroupType(node1->glyphType() ) == gar::ggVariant) {
        if(!isNodeVariable(node0->glyphType() ) ) {
            qDebug() << " variation input should be sprite or cylinder, rejected ";
            return true;
        }
    }
    return false;
}

GardenGlyph* GardenConnection::node0() const
{ return PortToNode(port0()); }

GardenGlyph* GardenConnection::node1() const
{ return PortToNode(port1()); }

GardenGlyph * GardenConnection::PortToNode(const GlyphPort * pt)
{ return static_cast<GardenGlyph *>(pt->parentItem() ); }

