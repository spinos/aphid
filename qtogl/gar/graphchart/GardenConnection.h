/*
 *  GardenConnection.h
 *
 *  with can make connection check
 *
 *  Created by jian zhang on 4/1/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef GARDENCONNECTION_H
#define GARDENCONNECTION_H

#include <qt/GlyphConnection.h>

namespace aphid {

class GlyphPort;
}

class GardenGlyph;
  
class  GardenConnection : public aphid::GlyphConnection
{

public:
    GardenConnection(QGraphicsItem * parent = 0);
	virtual ~GardenConnection();
	
	virtual bool canConnectTo(aphid::GlyphPort* p1) const;

	GardenGlyph* node0() const;
	GardenGlyph* node1() const;
	
private:
	bool isNodeVariable(int gt) const;
    bool rejectedByNode(GardenGlyph* node0, GardenGlyph* node1) const;
	static GardenGlyph * PortToNode(const aphid::GlyphPort * pt);
	
};

#endif        //  #ifndef APH_GGARDENCONNECTION_H

