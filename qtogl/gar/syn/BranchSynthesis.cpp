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

using namespace aphid;

namespace gar {

BranchSynthesis::BranchSynthesis()
{}

BranchSynthesis::~BranchSynthesis()
{}

BranchingProfile* BranchSynthesis::profile()
{ return &m_profile; }

bool BranchSynthesis::synthesizeAGroup(PieceAttrib* stemAttr)
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
	rootBlock->tmR()->setIdentity();
	rootBlock->setGeomInd(selprof._index);
	rootBlock->setExclR(selprof._exclR);
	
	m_blockStack.push(rootBlock);

/// first branching
	gar::SelectBudContext selbud;
	selbud._variationIndex = rootBlock->geomInd();
	selbud._budType = gar::bdLateral;
	rootBlock->worldTm().glMatrix(selbud._relMat);
	stemAttr->selectBud(&selbud);
	
	growOnLateralBud(&selbud, stemAttr, rootBlock);
	
	BlockPtrType curBlock;
	bool hasGrowth = m_blockStack.pop(curBlock);
	while (hasGrowth) {
		growOnStem(stemAttr, curBlock);
		
		hasGrowth = m_blockStack.pop(curBlock);
	}
	
	gar::SynthesisGroup* gi = addSynthesisGroup();
/// instance of root stem
	gi->addInstance(rootBlock->geomInd(), rootBlock->worldTm() );
/// todo exclR of whole branch
	gi->setExclusionRadius(rootBlock->exclR() );
	
	addBlockChildInstance(gi, rootBlock);
	
	return true;
}

void BranchSynthesis::growOnStem(PieceAttrib* stemAttr,
						StemBlock* parentStem)
{
	if(parentStem->age() > m_profile._numSeasons)
		return;
		
	gar::SelectBudContext selbud;
	selbud._upLimit = 0.f;
	selbud._variationIndex = parentStem->geomInd();
	parentStem->worldTm().glMatrix(selbud._relMat);
	stemAttr->selectBud(&selbud);
	BlockPtrType stm = growOnTerminalBud(&selbud, stemAttr, parentStem);
	
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
	selprof._condition = gar::slCloseToUp;
	childBlock->worldTm().glMatrix(selprof._relMat);
	selprof._upVec[0] = RandomFn11() * .2f;
	selprof._upVec[2] = RandomFn11() * .2f;
	ATriangleMesh* inGeom = stemAttr->selectGeom(&selprof);
	
	childBlock->setGeomInd(selprof._index);
	childBlock->setExclR(selprof._exclR);
	
	m_blockStack.push(childBlock);
	
	return childBlock;
}

void BranchSynthesis::growOnLateralBud(gar::SelectBudContext* selbud,
						PieceAttrib* stemAttr,
						StemBlock* parentStem)
{	
	const int& n = selbud->_numSelect;
	if(n < 1)
		return;
		
	int nb = 1 + rand() & 3;
	if(nb > n)
		nb = n;
		
	for(int i = 0;i<nb;++i) {
	
	Matrix44F budTm(selbud->_budTm[i]);
	
	BlockPtrType childBlock = new StemBlock(parentStem);
	*childBlock->tmR() = budTm;
	
	gar::SelectProfile selprof;
	selprof._condition = gar::slRandom;
	ATriangleMesh* inGeom = stemAttr->selectGeom(&selprof);
	
	childBlock->setGeomInd(selprof._index);
	childBlock->setExclR(selprof._exclR);
	childBlock->updateWorldTm();
	
	m_blockStack.push(childBlock);
	}
}

void BranchSynthesis::addBlockChildInstance(gar::SynthesisGroup* g,
					StemBlock* parentStem)
{	
	const int n = parentStem->numChildBlocks();
	for(int i=0;i<n;++i) {
		StemBlock* childStem = parentStem->childStem(i);
		
		addBlockChildInstance(g, childStem);
		
		g->addInstance(childStem->geomInd(), childStem->worldTm() );
		
	}
}

}
