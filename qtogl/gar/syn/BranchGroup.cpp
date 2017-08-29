/*
 *  BranchGroup.cpp
 * 
 *  synthesis group with root 
 *
 *  Created by jian zhang on 8/16/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "BranchGroup.h"
#include <math/Matrix44F.h>
#include "StemBlock.h"
#include <math/miscfuncs.h>

using namespace aphid;

namespace gar {

BranchGroup::BranchGroup(StemBlock* root)
{ m_rootBlock = root; }

BranchGroup::~BranchGroup()
{ delete m_rootBlock; }

void BranchGroup::addBlockInstances()
{
/// instance of root stem
	addInstance(m_rootBlock->geomInstanceInd(), m_rootBlock->worldTm() );

	setExclusionRadius(m_rootBlock->exclR() );
	addBlockChildInstance(m_rootBlock);	
}

void BranchGroup::addBlockChildInstance(StemBlock* parentStem)
{	
	const int n = parentStem->numChildBlocks();
	for(int i=0;i<n;++i) {
		StemBlock* childStem = parentStem->childStem(i);
		addBlockChildInstance(childStem);
		
		addInstance(childStem->geomInstanceInd(), childStem->worldTm() );
		
	}
}

void BranchGroup::calculateExclusionRadius()
{
	setExclusionRadius(1.f);
	Vector3F meanDisplace(0.f, 0.f, 0.f);
	int count = 0;
	addStemBlockDisplace(meanDisplace, count, m_rootBlock);
	meanDisplace /= (float)count;
	
	Matrix44F invspace;
	Matrix33F rotm;
	if(rotm.rotateUpTo(meanDisplace) ) {
		invspace.setRotation(rotm);
		invspace.inverse();
	}
	
	calculateDisplaceR(meanDisplace, invspace, m_rootBlock);
}

void BranchGroup::addStemBlockDisplace(Vector3F& dest, int& count, StemBlock* parentBlock)
{
	if(!parentBlock->isStem() )
		return;
		
	dest += parentBlock->worldTm().getTranslation();
	count++;
	
	const int n = parentBlock->numChildBlocks();	
	for(int i=0;i<n;++i) {
		StemBlock* childStem = parentBlock->childStem(i);
		addStemBlockDisplace(dest, count, childStem);
		
	}
}

void BranchGroup::calculateDisplaceR(const Vector3F& refVec, 
				const Matrix44F& invMat, 
				StemBlock* parentBlock)
{
	if(!parentBlock->isStem() )
		return;
		
	Vector3F displace = parentBlock->worldTm().getTranslation();
/// rotate to man
	displace = refVec + (displace - refVec) * .57f;
	
	displace = invMat.transform(displace);
/// project to x-z plane
	displace.y = 0.f;
	
	adjustExclusionRadius(displace.length() );
	
	const int n = parentBlock->numChildBlocks();	
	for(int i=0;i<n;++i) {
		StemBlock* childStem = parentBlock->childStem(i);
		calculateDisplaceR(refVec, invMat, childStem);
		
	}
}

}