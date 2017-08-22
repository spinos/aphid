/*
 *  BlockDeformAttribs.cpp
 *  
 *
 *  Created by jian zhang on 8/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "BlockDeformAttribs.h"
#include <geom/ATriangleMesh.h>
#include <geom/BlockDeformer.h>
#include <math/miscfuncs.h>
#include <gar_common.h>

using namespace aphid;

int BlockDeformAttribs::sNumInstances = 0;

BlockDeformAttribs::BlockDeformAttribs() : PieceAttrib(gar::gtBendTwistRollVariant),
m_inAttr(NULL),
m_inGeom(NULL),
m_exclR(1.f)
{
    m_instId = sNumInstances;
	sNumInstances++;
	
	addVector2Attrib(gar::nBend, 0.f, 0.5f);
	addFloatAttrib(gar::nTwist, 0.1f, 0.f, 1.f);
	addFloatAttrib(gar::nRoll, 0.1f, 0.f, 1.f);
	addSplineAttrib(gar::nWeightVariation);
	
	m_dfm = new BlockDeformer;	
	for(int i=0;i<32;++i) 
	    m_outGeom[i] = new ATriangleMesh;
}

void BlockDeformAttribs::setInputGeom(ATriangleMesh* x)
{ m_inGeom = x; }

bool BlockDeformAttribs::hasGeom() const
{ return m_inGeom != NULL; }

int BlockDeformAttribs::numGeomVariations() const
{ return 32; }

ATriangleMesh* BlockDeformAttribs::selectGeom(gar::SelectProfile* prof) const
{
	if(prof->_condition != gar::slIndex)
		prof->_index = rand() % numGeomVariations();
		
    prof->_exclR = m_exclR;
	prof->_height = m_geomHeight;
	return m_outGeom[prof->_index]; 
}

bool BlockDeformAttribs::update()
{    
    if(!m_inAttr)
        return false;
    
	gar::SelectProfile selprof;
    m_inGeom = m_inAttr->selectGeom(&selprof);
    
    if(!m_inGeom)
        return false;
		
	m_exclR = selprof._exclR;
	m_geomHeight = selprof._height;
		
	BlockDeformerBuilder builder;
/// a chain of 4
	float deltaHeight = m_geomHeight * .25;
	deform::Block* lastBlock = NULL;
	for(int i=0;i<4;++i) {
		deform::Block* curBlock = new deform::Block(lastBlock);
/// local translation of each block
		Matrix44F* ltm = curBlock->tmR();
		ltm->setTranslation(0.f, deltaHeight, 0.f);
		
		builder.addBlock(curBlock);
		lastBlock = curBlock;
	}
	m_dfm->createBlockDeformer(m_inGeom, builder);
	
    float bendRange[2];
	findAttrib(gar::nBend)->getValue2(bendRange);
	
	float twistRoll[2];
	findAttrib(gar::nTwist)->getValue(twistRoll[0]);
	findAttrib(gar::nRoll)->getValue(twistRoll[1]);
	
/// 4 bend/roll groups
	const float deltaBend = (bendRange[1] - bendRange[0]) * .25f;
	const float deltaRoll = twistRoll[1] * .25f;
		
	float angles[3];
    for(int i=0;i<32;++i) {
        int bendGrp = i>>3;
        angles[0] = bendRange[0] + deltaBend * (RandomF01() + bendGrp);
        angles[1] = twistRoll[0] * (.5f + RandomF01() * .5f);
		if(i&1) {
            angles[1] = -angles[1];
		}
/// roll distribution
        angles[2] = deltaRoll * ((float)i - (bendGrp<<3) - 4 + RandomF01() );

		m_dfm->setBend(angles[0]);
		m_dfm->setTwist(angles[1]);
		m_dfm->setRoll(angles[2]);
		m_dfm->deform(m_inGeom);
		m_dfm->updateGeom(m_outGeom[i], m_inGeom);
	}
	
	computeTexcoord(m_outGeom, 32, m_inAttr->texcoordBlockAspectRatio() );
	
	return true;
}

int BlockDeformAttribs::attribInstanceId() const
{ return m_instId; }

void BlockDeformAttribs::connectTo(PieceAttrib* another, const std::string& portName)
{
    if(!another->hasGeom()) {
        std::cout<<"\n ERROR BlockDeformAttribs cannot connect input geom ";
        m_inGeom = NULL;  
        return;
    }
    
    m_inAttr = another;
    update();
}

bool BlockDeformAttribs::isGeomStem() const
{
	if(!m_inAttr)
		return false;
	return m_inAttr->isGeomStem();
}

bool BlockDeformAttribs::isGeomLeaf() const
{
	if(!m_inAttr)
		return false;
	return m_inAttr->isGeomLeaf();
}

bool BlockDeformAttribs::canConnectToViaPort(const PieceAttrib* another, const std::string& portName) const
{
	return (another->isGeomStem() || another->isGeomLeaf() );
}
