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
#include <math/SplineMap1D.h>

class PieceAttrib;
class StemBlock;

namespace gar {

struct SelectBudContext;
struct SelectProfile;

struct BranchingProfile {
	int _numSeasons;
/// begin and end
	int _axialSeason[2];
/// min and max
	int _numLateralShoots[2];
/// begin of foliage
	int _leafSeason;
	float _axil;
/// sloping of ground affects up direction
	float _tilt;
	float _upVec[3];
/// of stem
	float _ascending;
	aphid::SplineMap1D _ascendVaring;
	
	BranchingProfile()
	{
		_numSeasons = 4;
		_axialSeason[0] = 0;
		_axialSeason[1] = 2;
		_numLateralShoots[0] = 1;
		_numLateralShoots[1] = 3;
		_leafSeason = 2;
		_ascending = .2f;
		_axil = 1.2f;
		_tilt = 0.f;
		_upVec[0] = 0.f;
		_upVec[1] = 1.f;
		_upVec[2] = 0.f;
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
	bool synthesizeAGroup(PieceAttrib* stemAttr,
					PieceAttrib* leafAttr);
	
private:
	void calculateUpVec();
	void getUpVecWithNoise(float* dest, const float& noiwei) const;
	void growOnStem(PieceAttrib* stemAttr, 
					StemBlock* parentStem);
/// growth of this season
	BlockPtrType growOnTerminalBud(gar::SelectBudContext* selbud,
						PieceAttrib* stemAttr,
						StemBlock* parentStem);
	void growOnLateralBud(gar::SelectBudContext* selbud,
						PieceAttrib* stemAttr,
						StemBlock* parentStem);						
	void growLeafOnStem(PieceAttrib* stemAttr, PieceAttrib* leafAttr,
						StemBlock* parentStem);
	void growTerminalLeaf(PieceAttrib* stemAttr, PieceAttrib* leafAttr,
						StemBlock* parentStem);
	void growLateralLeaf(PieceAttrib* stemAttr, PieceAttrib* leafAttr,
						StemBlock* parentStem);	
	
};

}

#endif
