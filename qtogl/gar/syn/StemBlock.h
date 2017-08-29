/*
 *  StemBlock.h
 *  
 *
 *  Created by jian zhang on 8/24/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_STEM_BLOCK_H
#define GAR_STEM_BLOCK_H

#include <geom/BlockDeformer.h>

class StemBlock : public aphid::deform::Block {

	int m_age;
	int m_geomInd;
	float m_exclR;
	bool m_hasTerminalStem;
	
	enum BlockType {
		tUnknown = 0,
		tLateralStem,
		tAxialStem,
		tLeaf,
	};
	BlockType m_blkType;
	
public:
	StemBlock(StemBlock* parent = 0);
	
	void setGeomInd(int x);
	void setExclR(float x);
	void setIsLateral();
	void setIsAxial();
	void setIsLeaf();
	void setHasTerminalStem();

	const int& age() const;
	const int& geomInd() const;
	const float& exclR() const;
	bool isLateral() const;
	bool isAxial() const;
	bool isLeaf() const;
	bool isStem() const;
	const bool& hasTerminalStem() const;
	
	StemBlock* childStem(int i) const;
	
/// stem: (geomInd + 1)<<20
	int geomInstanceInd() const;
	
protected:

private:	
};

#endif
