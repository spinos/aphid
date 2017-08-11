/*
 *  BendTwistRollAttribs.cpp
 *  
 *
 *  Created by jian zhang on 8/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "BendTwistRollAttribs.h"
#include <geom/ATriangleMesh.h>

using namespace aphid;

int BendTwistRollAttribs::sNumInstances = 0;

BendTwistRollAttribs::BendTwistRollAttribs() :
m_inGeom(NULL)
{
    m_instId = sNumInstances;
	sNumInstances++;
	
	addFloatAttrib(gar::nBend, 1.f, 0.f, 1.f);
	addFloatAttrib(gar::nTwist, 0.5f, 0.f, 1.f);
	addFloatAttrib(gar::nRoll, 0.2f, 0.f, 1.f);
}

void BendTwistRollAttribs::setInputGeom(ATriangleMesh* x)
{ m_inGeom = x; }

bool BendTwistRollAttribs::hasGeom() const
{ return m_inGeom != NULL; }

int BendTwistRollAttribs::numGeomVariations() const
{ return 64; }

ATriangleMesh* BendTwistRollAttribs::selectGeom(int x, float& exclR) const
{
	return m_outGeom[x]; 
}

bool BendTwistRollAttribs::update()
{
    std::cout<<"\n BendTwistRollAttribs::update ";
        std::cout.flush();
    m_outGeom[0] = m_inGeom;
	return true;
}

int BendTwistRollAttribs::attribInstanceId() const
{ return m_instId; }

void BendTwistRollAttribs::connectTo(PieceAttrib* another)
{
    if(!another->hasGeom()) {
        std::cout<<"\n ERROR BendTwistRollAttribs cannot connect input geom ";
        m_inGeom = NULL;  
        return;
    }
    
    float r;
    m_inGeom = another->selectGeom(0, r);
    update();
}
