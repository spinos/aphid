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
m_inStemAttr(NULL)
{
    m_instId = sNumInstances;
	sNumInstances++;
	
	addActionAttrib(gar::nShuffle, ":/icons/shuffle.png");
	addIntAttrib(gar::nNumSeasons, 4, 1, 9);
	
}

SimpleBranchAttribs::~SimpleBranchAttribs()
{
}

bool SimpleBranchAttribs::update()
{    
    if(!m_inStemAttr)
        return false;
		
	gar::BranchingProfile* prof = profile();
	findAttrib(gar::nNumSeasons)->getValue(prof->_numSeasons);
	
	gar::Attrib* shuffleA = findAttrib(gar::nShuffle);
	int ishuffle;
	shuffleA->getValue(ishuffle);
	
	if(ishuffle > 0) {
		shuffleA->setValue(0);
		clearSynths();
	}
		
	if(numSynthesizedGroups() < 1)
		synthesizeAGroup(m_inStemAttr);
    
	return true;
}

int SimpleBranchAttribs::attribInstanceId() const
{ return m_instId; }

bool SimpleBranchAttribs::connectToStem(PieceAttrib* another)
{
	m_inStemAttr = another;
	return true;
}

bool SimpleBranchAttribs::canConnectToViaPort(const PieceAttrib* another, const std::string& portName) const
{
	return another->isGeomBranchingUnit();
}

void SimpleBranchAttribs::connectTo(PieceAttrib* another, const std::string& portName)
{
	bool stat = false;
	if(portName == "inStem") 
		stat = connectToStem(another);
		
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

ATriangleMesh* SimpleBranchAttribs::selectGeom(gar::SelectProfile* prof) const
{
	return selectStemGeom(prof);
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
