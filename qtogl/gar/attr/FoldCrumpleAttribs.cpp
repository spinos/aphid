/*
 *  FoldCrumpleAttribs.cpp
 *  
 *  unfolding, crumpling
 *  
 *  Created by jian zhang on 8/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "FoldCrumpleAttribs.h"
#include <geom/ATriangleMesh.h>
#include <geom/FoldCrumpleDeformer.h>
#include <math/miscfuncs.h>
#include <gar_common.h>

using namespace aphid;

int FoldCrumpleAttribs::sNumInstances = 0;

FoldCrumpleAttribs::FoldCrumpleAttribs() : PieceAttrib(gar::gtFoldCrumpleVariant),
m_inAttr(NULL),
m_inGeom(NULL),
m_exclR(1.f)
{
    m_instId = sNumInstances;
	sNumInstances++;
	
	addVector2Attrib(gar::nFold, .5f, -.5f);
	addFloatAttrib(gar::nmCrumple, 0.1f, 0.f, 1.f);
	addSplineAttrib(gar::nFoldVar);
	addSplineAttrib(gar::nCrumpleVar);
	addVector2Attrib(gar::nBend, 0.f, 0.5f);
	addFloatAttrib(gar::nTwist, 0.1f, 0.f, 1.f);
	addFloatAttrib(gar::nRoll, 0.1f, 0.f, 1.f);
	addSplineAttrib(gar::nWeightVariation);
	
	m_dfm = new FoldCrumpleDeformer;
	for(int i=0;i<48;++i) 
	    m_outGeom[i] = new ATriangleMesh;
}

FoldCrumpleAttribs::~FoldCrumpleAttribs()
{}

void FoldCrumpleAttribs::setInputGeom(ATriangleMesh* x)
{ m_inGeom = x; }

bool FoldCrumpleAttribs::hasGeom() const
{ return m_inGeom != NULL; }

int FoldCrumpleAttribs::numGeomVariations() const
{ return 48; }

ATriangleMesh* FoldCrumpleAttribs::selectGeom(gar::SelectProfile* prof) const
{
	if(prof->_condition != gar::slIndex)
		prof->_index = rand() % numGeomVariations();
		
    prof->_exclR = m_exclR;
	return m_outGeom[prof->_index]; 
}

bool FoldCrumpleAttribs::update()
{    
    if(!m_inAttr)
        return false;
    
	gar::SelectProfile selprof;
	selprof._condition = gar::slIndex;
	selprof._index = 0;
    m_inGeom = m_inAttr->selectGeom(&selprof);
    
    if(!m_inGeom)
        return false;
		
	m_exclR = selprof._exclR;
		
	float bendRange[2];
	findAttrib(gar::nBend)->getValue2(bendRange);
	
	float twistRoll[2];
	findAttrib(gar::nTwist)->getValue(twistRoll[0]);
	findAttrib(gar::nRoll)->getValue(twistRoll[1]);
	
	float foldRange[2];
	findAttrib(gar::nFold)->getValue2(foldRange);
	float crumpleRng = 0;
	findAttrib(gar::nmCrumple)->getValue(crumpleRng);
	
	SplineMap1D* ws = m_dfm->weightSpline();
	gar::SplineAttrib* aws = (gar::SplineAttrib*)findAttrib(gar::nWeightVariation);
	updateSplineValues(ws, aws);
	
	SplineMap1D* fs = m_dfm->foldSpline();
	gar::SplineAttrib* afs = (gar::SplineAttrib*)findAttrib(gar::nFoldVar);
	updateSplineValues(fs, afs);
	
	SplineMap1D* cs = m_dfm->crumpleSpline();
	gar::SplineAttrib* acs = (gar::SplineAttrib*)findAttrib(gar::nCrumpleVar);
	updateSplineValues(cs, acs);
	
	m_dfm->computeRowWeight(m_inGeom);
	
/// 8 bend/fold/crumple groups
	const float deltaBend = (bendRange[1] - bendRange[0]) * .125f;
	const float deltaFold = (foldRange[1] - foldRange[0]) * .125f;
	const float deltaCrumple = crumpleRng * .125f;
	
/// 6 twist/roll per group
	const float deltaTwist = twistRoll[0] * .33f;
	const float deltaRoll = twistRoll[1] * .33f;
		
	for(int j=0;j<8;++j) {
		float bendF = bendRange[0] + deltaBend * j;
        float foldF = foldRange[0] + deltaFold * j;
		float crumpleF = deltaCrumple * j;
		
        for(int i=0;i<6;++i) {
			m_dfm->setBend(bendF + deltaBend * RandomF01() );
			m_dfm->setFold(foldF + deltaFold * RandomF01() );
			m_dfm->setCrumple(crumpleF + deltaCrumple * RandomF01() );
			
			m_dfm->setTwist(-twistRoll[0] + deltaTwist * ((float)i + RandomF01() ) );
			m_dfm->setRoll(-twistRoll[1] + deltaRoll * ((float)i + RandomF01() ) );
			
			m_dfm->deform(m_inGeom);
			m_dfm->updateGeom(m_outGeom[j * 6 +i], m_inGeom);
		}
	}
	
	computeTexcoord(m_outGeom, 48, m_inAttr->texcoordBlockAspectRatio() );
	
	return true;
}

int FoldCrumpleAttribs::attribInstanceId() const
{ return m_instId; }

void FoldCrumpleAttribs::connectTo(PieceAttrib* another, const std::string& portName)
{
    if(!another->hasGeom()) {
        std::cout<<"\n ERROR FoldCrumpleAttribs cannot connect input geom ";
        m_inGeom = NULL;  
        return;
    }
    
    m_inAttr = another;
    update();
}

bool FoldCrumpleAttribs::isGeomStem() const
{
	if(!m_inAttr)
		return false;
	return m_inAttr->isGeomStem();
}

bool FoldCrumpleAttribs::isGeomLeaf() const
{
	if(!m_inAttr)
		return false;
	return m_inAttr->isGeomLeaf();
}

bool FoldCrumpleAttribs::canConnectToViaPort(const PieceAttrib* another, const std::string& portName) const
{
	return another->isGeomLeaf();
}
