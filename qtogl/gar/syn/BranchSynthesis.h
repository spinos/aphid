/*
 *  BranchSynthesis.h
 *  
 *
 *  Created by jian zhang on 8/24/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_BRANCH_SYNTHESIS_H
#define GAR_BRANCH_SYNTHESIS_H

#include "MultiSynthesis.h"
#include <foundation/BoundedStack.h>

class PieceAttrib;
class StemBlock;

namespace gar {

struct SelectBudContext;
struct SelectProfile;

struct BranchingProfile {
	int _numSeasons;
	
	BranchingProfile()
	{
		_numSeasons = 4;
	}
	
};

class BranchSynthesis : public MultiSynthesis {

typedef StemBlock* BlockPtrType;
	aphid::BoundedStack<BlockPtrType, 128> m_blockStack;
	
	BranchingProfile m_profile;
	
public:
	BranchSynthesis();
	virtual ~BranchSynthesis();
	
protected:
	BranchingProfile* profile();
	bool synthesizeAGroup(PieceAttrib* stemAttr);
	
private:
	void growOnStem(PieceAttrib* stemAttr, 
					StemBlock* parentStem);
	void addBlockChildInstance(gar::SynthesisGroup* g,
					StemBlock* parentStem);
/// growth of this season
	BlockPtrType growOnTerminalBud(gar::SelectBudContext* selbud,
						PieceAttrib* stemAttr,
						StemBlock* parentStem);
	void growOnLateralBud(gar::SelectBudContext* selbud,
						PieceAttrib* stemAttr,
						StemBlock* parentStem);
};

}

#endif
