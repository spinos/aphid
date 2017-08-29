/*
 *  BranchGroup.h
 *  
 *  synthesis group with root 
 *
 *  Created by jian zhang on 8/16/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_BRANCH_GROUP_H
#define GAR_BRANCH_GROUP_H

#include "SynthesisGroup.h"

namespace aphid {
class Vector3F;
class Matrix44F;
}

class StemBlock;

namespace gar {

class BranchGroup : public SynthesisGroup {
	
	StemBlock* m_rootBlock;
	
public:
	BranchGroup(StemBlock* root);
	~BranchGroup();
	
	void addBlockInstances();
/// for each terminal block, adjust exclR by half x-z r
	void calculateExclusionRadius();
	
protected:

private:
	void addBlockChildInstance(StemBlock* parentStem);
	void addStemBlockDisplace(aphid::Vector3F& dest, int& count, 
				StemBlock* parentBlock);
	void calculateDisplaceR(const aphid::Vector3F& refVec, 
				const aphid::Matrix44F& invMat,
				StemBlock* parentBlock);
};

}

#endif