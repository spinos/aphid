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

class StemBlock;

namespace gar {

class BranchGroup : public SynthesisGroup {
	
	StemBlock* m_rootBlock;
	
public:
	BranchGroup(StemBlock* root);
	~BranchGroup();
	
	void addBlockInstances();
	
protected:

private:
	void addBlockChildInstance(StemBlock* parentStem);
	
};

}

#endif