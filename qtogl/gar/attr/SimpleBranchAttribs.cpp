/*
 *  SimpleBranchAttribs.cpp
 *  
 *  synthesize from a stem and many leaves
 *
 *  Created by jian zhang on 8/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "SimpleBranchAttribs.h"
#include <geom/ATriangleMesh.h>
#include <math/miscfuncs.h>
#include <math/SplineMap1D.h>
#include <gar_common.h>
#include <syn/SynthesisGroup.h>

using namespace aphid;

int SimpleBranchAttribs::sNumInstances = 0;

SimpleBranchAttribs::SimpleBranchAttribs() : PieceAttrib(gar::gtSimpleBranch),
m_inStemAttr(NULL),
m_inLeafAttr(NULL)
{
    m_instId = sNumInstances;
	sNumInstances++;
	
	addActionAttrib(gar::nShuffle, ":/icons/shuffle.png");
	addIntAttrib(gar::nGrowSeasons, 4, 1, 10);
	addInt2Attrib(gar::nAxialSeasons, 0, 2);
	addInt2Attrib(gar::nLateralShoots, 1, 3);
	addFloatAttrib(gar::nAscendAngle, .2f, 0.f, .5f);
	addSplineAttrib(gar::nAscendVary);
	addFloatAttrib(gar::nTilt, 0.f, 0.f, 1.f);
	addIntAttrib(gar::nLeafSeason, 3, 1, 10);
	addFloatAttrib(gar::nAxil, 1.2f, 0.4f, 2.f);
	
	gar::SplineAttrib* acs = (gar::SplineAttrib*)findAttrib(gar::nAscendVary);
	acs->setSplineValue(.5f, .5f);
	acs->setSplineCv0(.4f, .5f);
	acs->setSplineCv1(.6f, .5f);
}

SimpleBranchAttribs::~SimpleBranchAttribs()
{}

bool SimpleBranchAttribs::update()
{    
    if(!m_inStemAttr)
        return false;
		
	gar::BranchingProfile* prof = profile();
	findAttrib(gar::nGrowSeasons)->getValue(prof->_numSeasons);
	findAttrib(gar::nAxialSeasons)->getValue2(prof->_axialSeason);
	findAttrib(gar::nLateralShoots)->getValue2(prof->_numLateralShoots);
	findAttrib(gar::nAscendAngle)->getValue(prof->_ascending);
	findAttrib(gar::nLeafSeason)->getValue(prof->_leafSeason);
	findAttrib(gar::nAxil)->getValue(prof->_axil);
	findAttrib(gar::nTilt)->getValue(prof->_tilt);
	
	SplineMap1D* acs = &prof->_ascendVaring;
	gar::SplineAttrib* aacs = (gar::SplineAttrib*)findAttrib(gar::nAscendVary);
	updateSplineValues(acs, aacs);
	
	gar::Attrib* shuffleA = findAttrib(gar::nShuffle);
	int ishuffle;
	shuffleA->getValue(ishuffle);
	
	if(ishuffle > 0) {
		shuffleA->setValue(0);
		clearSynths();
	}
		
	if(numSynthesizedGroups() < 1)
		synthesizeAGroup(m_inStemAttr, m_inLeafAttr);
    
	return true;
}

int SimpleBranchAttribs::attribInstanceId() const
{ return m_instId; }

bool SimpleBranchAttribs::connectToStem(PieceAttrib* another)
{
	m_inStemAttr = another;
	return true;
}

bool SimpleBranchAttribs::connectToLeaf(PieceAttrib* another)
{
	m_inLeafAttr = another;
	return true;
}

bool SimpleBranchAttribs::canConnectToViaPort(const PieceAttrib* another, const std::string& portName) const
{
	if(portName == "inStem") 
		return another->isGeomBranchingUnit();
/// leaf or twig
	return another->isGeomLeaf() || another->isTwig();
}

void SimpleBranchAttribs::connectTo(PieceAttrib* another, const std::string& portName)
{
	bool stat = false;
	if(portName == "inStem") 
		stat = connectToStem(another);
	else
		stat = connectToLeaf(another);
		
    if(!stat) {
        std::cout<<"\n ERROR SimpleBranchAttribs cannot connect input attr ";
        return;
    }
    
    update();
}

ATriangleMesh* SimpleBranchAttribs::selectStemGeom(gar::SelectProfile* prof) const
{
    if(!m_inStemAttr)
		return NULL;
		
/// recover stem geom ind
	prof->_index = (prof->_index>>20) - 1;
#if 0	
		std::cout<<"\n stem attr"<<m_inStemAttr->glyphType()
		<<" instance"<<m_inStemAttr->attribInstanceId()
		<<" geom"<<prof->_index;
		std::cout.flush();
#endif
		
	prof->_geomInd = (gar::GlyphTypeToGeomIdGroup(m_inStemAttr->glyphType() ) 
	                    | (m_inStemAttr->attribInstanceId() << 10) 
	                    | prof->_index);
		
	return m_inStemAttr->selectGeom(prof);
}

ATriangleMesh* SimpleBranchAttribs::selectLeafGeom(gar::SelectProfile* prof) const
{
    if(!m_inLeafAttr)
		return NULL;
		
	return m_inLeafAttr->selectGeom(prof);
}

ATriangleMesh* SimpleBranchAttribs::selectGeom(gar::SelectProfile* prof) const
{
	if(prof->_index > 1048575)
		return selectStemGeom(prof);
		
	return selectLeafGeom(prof);
}

bool SimpleBranchAttribs::isSynthesized() const
{ return true; }

int SimpleBranchAttribs::numSynthesizedGroups() const
{ return synthsisGroups().size(); }

gar::SynthesisGroup* SimpleBranchAttribs::selectSynthesisGroup(gar::SelectProfile* prof) const
{
	if(prof->_condition != gar::slIndex)
		prof->_index = rand() % numSynthesizedGroups();
	
	prof->_exclR = synthsisGroups()[prof->_index]->exclusionRadius();	
	return synthsisGroups()[prof->_index];
}
