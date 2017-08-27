/*
 *  BranchSynthesis.cpp
 *  
 *
 *  Created by jian zhang on 8/24/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "BranchSynthesis.h"
#include "SynthesisGroup.h"
#include <attr/PieceAttrib.h>
#include <geom/ATriangleMesh.h>
#include "StemBlock.h"
#include <math/miscfuncs.h>
#include "BranchGroup.h"

using namespace aphid;

namespace gar {

BranchSynthesis::BranchSynthesis()
{}

BranchSynthesis::~BranchSynthesis()
{}

BranchingProfile* BranchSynthesis::profile()
{ return &m_profile; }

bool BranchSynthesis::synthesizeAGroup(PieceAttrib* stemAttr,
								PieceAttrib* leafAttr)
{
	gar::SelectProfile selprof;
	selprof._condition = gar::slCloseToUp;
	Matrix44F::IdentityMatrix.glMatrix(selprof._relMat);
	selprof._upVec[0] = RandomFn11() * .2f;
	selprof._upVec[2] = RandomFn11() * .2f;
	
	ATriangleMesh* inGeom = stemAttr->selectGeom(&selprof);
	if(!inGeom)
		return false;
		
	m_blockStack.clear();
	
	BlockPtrType rootBlock = new StemBlock;
	rootBlock->setIsAxial();
	rootBlock->tmR()->setIdentity();
	rootBlock->setGeomInd(selprof._index);
	rootBlock->setExclR(selprof._exclR);
	
	m_blockStack.push(rootBlock);

	if(m_profile._axialSeason[0] == 0) {
/// first branching
		gar::SelectBudContext selbud;
		selbud._ascending = m_profile._ascending;
		selbud._ascendVaring = .5f * (m_profile._ascendVaring.interpolate(0.f) - .5f);
		selbud._variationIndex = rootBlock->geomInd();
		selbud._budType = gar::bdLateral;
		rootBlock->worldTm().glMatrix(selbud._relMat);
		stemAttr->selectBud(&selbud);
		
		growOnLateralBud(&selbud, stemAttr, rootBlock);
	}
	
	BlockPtrType curBlock;
	bool hasGrowth = m_blockStack.pop(curBlock);
	while (hasGrowth) {
		growOnStem(stemAttr, curBlock);
		
		hasGrowth = m_blockStack.pop(curBlock);
	}
	
	growLeafOnStem(stemAttr, leafAttr, rootBlock);
	
	gar::BranchGroup* gi = new gar::BranchGroup(rootBlock);
	addSynthesisGroup(gi);
	
	gi->addBlockInstances();
	gi->calculateExclusionRadius();

	return true;
}

void BranchSynthesis::growOnStem(PieceAttrib* stemAttr,
						StemBlock* parentStem)
{
	if(parentStem->age() > m_profile._numSeasons)
		return;
		
	if(parentStem->isAxial() ) {
		if(parentStem->age() > m_profile._axialSeason[1])
			return;
	}
		
	gar::SelectBudContext selbud;
	selbud._ascending = m_profile._ascending;
	const float ageparam = (float)parentStem->age() / (float)m_profile._numSeasons;
	selbud._ascendVaring = .5f * (m_profile._ascendVaring.interpolate(ageparam) - .5f);
	selbud._upLimit = 0.f;
	selbud._variationIndex = parentStem->geomInd();
	parentStem->worldTm().glMatrix(selbud._relMat);
	stemAttr->selectBud(&selbud);
	BlockPtrType stm = growOnTerminalBud(&selbud, stemAttr, parentStem);
	
	if(parentStem->isAxial() ) {
		if(parentStem->age() < m_profile._axialSeason[0])
			return;
	}
	
	selbud._variationIndex = stm->geomInd();
	selbud._budType = gar::bdLateral;
	stm->worldTm().glMatrix(selbud._relMat);
	stemAttr->selectBud(&selbud);
	growOnLateralBud(&selbud, stemAttr, stm);
	
}

BranchSynthesis::BlockPtrType BranchSynthesis::growOnTerminalBud(gar::SelectBudContext* selbud,
						PieceAttrib* stemAttr,
						StemBlock* parentStem)
{
	Matrix44F budTm(selbud->_budTm[0]);
		
	BlockPtrType childBlock = new StemBlock(parentStem);
	*childBlock->tmR() = budTm;
	childBlock->updateWorldTm();
	
	gar::SelectProfile selprof;
	if(parentStem->isAxial() ) {
		selprof._condition = gar::slCloseToUp;
		childBlock->worldTm().glMatrix(selprof._relMat);
		selprof._upVec[0] = RandomFn11() * .2f;
		selprof._upVec[2] = RandomFn11() * .2f;
		childBlock->setIsAxial();
		
	} else {
		selprof._condition = gar::slRandom;
	}
	
	ATriangleMesh* inGeom = stemAttr->selectGeom(&selprof);
	
	childBlock->setGeomInd(selprof._index);
	childBlock->setExclR(selprof._exclR);
	
	m_blockStack.push(childBlock);
	
	parentStem->setHasTerminalStem();
	
	return childBlock;
}

void BranchSynthesis::growOnLateralBud(gar::SelectBudContext* selbud,
						PieceAttrib* stemAttr,
						StemBlock* parentStem)
{	
	const int& n = selbud->_numSelect;
	if(n < 1)
		return;
		
	int nb = m_profile._numLateralShoots[0];
	if(m_profile._numLateralShoots[1] != m_profile._numLateralShoots[0])
		nb += RandomF01() * (1 + m_profile._numLateralShoots[1] - m_profile._numLateralShoots[0]);
		
	if(nb > n)
		nb = n;
	
	gar::SelectProfile selprof;
	selprof._condition = gar::slRandom;
	
	for(int i = 0;i<nb;++i) {
	
	Matrix44F budTm(selbud->_budTm[i]);
	
	BlockPtrType childBlock = new StemBlock(parentStem);
	*childBlock->tmR() = budTm;
	childBlock->updateWorldTm();
	
	ATriangleMesh* inGeom = stemAttr->selectGeom(&selprof);
	
	childBlock->setGeomInd(selprof._index);
	childBlock->setExclR(selprof._exclR);
	
	m_blockStack.push(childBlock);
	}
}

void BranchSynthesis::growLeafOnStem(PieceAttrib* stemAttr, PieceAttrib* leafAttr,
						StemBlock* parentStem)
{
	if(!leafAttr)
		return;
		
	const int n = parentStem->numChildBlocks();
	for(int i=0;i<n;++i) {
		StemBlock* childStem = parentStem->childStem(i);
		growLeafOnStem(stemAttr, leafAttr, childStem);
		
	}
	
	if(parentStem->age() < m_profile._leafSeason)
		return;
	
	if(!parentStem->hasTerminalStem() )
		growTerminalLeaf(stemAttr, leafAttr, parentStem);
		
	growLateralLeaf(stemAttr, leafAttr, parentStem);
	
}

void BranchSynthesis::growTerminalLeaf(PieceAttrib* stemAttr, PieceAttrib* leafAttr,
						StemBlock* parentStem)
{
	gar::SelectBudContext selbud;
	const float ageparam = (float)parentStem->age() / (float)m_profile._numSeasons;
	selbud._variationIndex = parentStem->geomInd();
	parentStem->worldTm().glMatrix(selbud._relMat);
	selbud._budType = gar::bdTerminalFoliage;
	stemAttr->selectBud(&selbud);
	
	Matrix44F budTm(selbud._budTm[0]);
		
	BlockPtrType childBlock = new StemBlock(parentStem);
	childBlock->setIsLeaf();
	*childBlock->tmR() = budTm;
	childBlock->updateWorldTm();
	
	gar::SelectProfile selprof;
	selprof._condition = gar::slRandom;
	
	ATriangleMesh* inGeom = leafAttr->selectGeom(&selprof);
	
	childBlock->setGeomInd(selprof._index);
	childBlock->setExclR(selprof._exclR);
	
}
	
void BranchSynthesis::growLateralLeaf(PieceAttrib* stemAttr, PieceAttrib* leafAttr,
						StemBlock* parentStem)
{
	gar::SelectBudContext selbud;
	selbud._ascending = m_profile._ascending;
	const float ageparam = (float)parentStem->age() / (float)m_profile._numSeasons;
	selbud._ascendVaring = .5f * (m_profile._ascendVaring.interpolate(ageparam) - .5f);
	selbud._upLimit = 0.f;
	selbud._variationIndex = parentStem->geomInd();
	parentStem->worldTm().glMatrix(selbud._relMat);
	selbud._budType = gar::bdLateralFoliage;
	selbud._condition = slAll;
	selbud._axil = m_profile._axil;
	stemAttr->selectBud(&selbud);

	const int& n = selbud._numSelect;
	if(n < 1)
		return;
	
	gar::SelectProfile selprof;
	selprof._condition = gar::slRandom;
		
	for(int i = 0;i<n;++i) {
	
		Matrix44F budTm(selbud._budTm[i]);
		
		BlockPtrType childBlock = new StemBlock(parentStem);
		childBlock->setIsLeaf();
		*childBlock->tmR() = budTm;
		childBlock->updateWorldTm();
		
		ATriangleMesh* inGeom = leafAttr->selectGeom(&selprof);
		
		childBlock->setGeomInd(selprof._index);
		childBlock->setExclR(selprof._exclR);
		
	}
}

}
