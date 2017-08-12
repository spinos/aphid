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
#include <geom/BendTwistRollDeformer.h>
#include <math/miscfuncs.h>

using namespace aphid;

int BendTwistRollAttribs::sNumInstances = 0;

BendTwistRollAttribs::BendTwistRollAttribs() :
m_inAttr(NULL),
m_inGeom(NULL),
m_exclR(1.f)
{
    m_instId = sNumInstances;
	sNumInstances++;
	
	addFloatAttrib(gar::nBend, 0.1f, 0.f, 1.f);
	addFloatAttrib(gar::nTwist, 0.1f, 0.f, 1.f);
	addFloatAttrib(gar::nRoll, 0.1f, 0.f, 1.f);
	
	m_dfm = new BendTwistRollDeformer;
	for(int i=0;i<32;++i) 
	    m_outGeom[i] = new ATriangleMesh;
}

void BendTwistRollAttribs::setInputGeom(ATriangleMesh* x)
{ m_inGeom = x; }

bool BendTwistRollAttribs::hasGeom() const
{ return m_inGeom != NULL; }

int BendTwistRollAttribs::numGeomVariations() const
{ return 32; }

ATriangleMesh* BendTwistRollAttribs::selectGeom(int x, float& exclR) const
{
    exclR = m_exclR;
	return m_outGeom[x]; 
}

bool BendTwistRollAttribs::update()
{    
    if(!m_inAttr)
        return false;
    
    m_inGeom = m_inAttr->selectGeom(0, m_exclR);
    
    if(!m_inGeom)
        return false;
    
    float rng[3];
	findAttrib(gar::nBend)->getValue(rng[0]);
	findAttrib(gar::nTwist)->getValue(rng[1]);
	findAttrib(gar::nRoll)->getValue(rng[2]);
	
/// 4 bend groups
	const float deltaBend = rng[0] * .25f;
		
	float angles[3];
    for(int i=0;i<32;++i) {
        int bendGrp = i>>3;
        angles[0] = deltaBend * (RandomF01() + bendGrp);
        angles[1] = rng[1] * (.5f + RandomF01() * .5f);
        if(i&1)
            angles[1] = -angles[1];
        
        angles[2] = rng[2] * RandomFn11();
        
        updateGeom(m_outGeom[i], angles);
	}
	
	computeTexcoord(m_outGeom, 32, m_inAttr->texcoordBlockAspectRatio() );
	
	return true;
}

void BendTwistRollAttribs::updateGeom(ATriangleMesh* msh,
        const float* angles)
{
    int np = m_inGeom->numPoints();
	int nt = m_inGeom->numTriangles();
	    
	msh->create(np, nt);
	msh->createVertexColors(np);
	
	unsigned * indDst = msh->indices();
	memcpy(indDst, m_inGeom->indices(), nt * 12);
	
	Vector3F * pntDst = msh->points();
	memcpy(pntDst, m_inGeom->points(), np * 12);
	
	Vector3F * nmlDst = msh->vertexNormals();
	memcpy(nmlDst, m_inGeom->vertexNormals(), np * 12);
	
	float * colDst = msh->vertexColors();
	memcpy(colDst, m_inGeom->vertexColors(), np * 12);
	
	float * texcoordDst = msh->triangleTexcoords();
	memcpy(texcoordDst, m_inGeom->triangleTexcoords(), nt * 24);
	
	m_dfm->setBend(angles[0]);
	m_dfm->setTwist(angles[1]);
	m_dfm->setRoll(angles[2]);
	m_dfm->deform(m_inGeom);
	
	memcpy(pntDst, m_dfm->deformedPoints(), np * 12);
	
	memcpy(nmlDst, m_dfm->deformedNormals(), np * 12);
	
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
    
    m_inAttr = another;
    update();
}
