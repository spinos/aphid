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

namespace gar {

BranchGroup::BranchGroup(StemBlock* root)
{ m_rootBlock = root; }

BranchGroup::~BranchGroup()
{ delete m_rootBlock; }

void BranchGroup::addBlockInstances()
{
/// instance of root stem
	addInstance(m_rootBlock->geomInstanceInd(), m_rootBlock->worldTm() );
/// todo exclR of whole branch
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

}