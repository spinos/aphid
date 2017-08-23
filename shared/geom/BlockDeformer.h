/*
 *  BlockDeformer.h
 *  
 *  deform a chain of blocks by bend x twist y roll z
 *
 *  Created by jian zhang on 8/10/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_BLOCK_DEFORMER_H
#define APH_BLOCK_DEFORMER_H

#include "TriangleMeshDeformer.h"
#include <math/Matrix44F.h>
#include <boost/scoped_array.hpp>
#include <vector>

namespace aphid {

namespace deform {

class Block {

	Matrix44F m_tm;
	Matrix44F m_wtm;
	Block* m_parentBlock;
	std::vector<Block* > m_childBlocks;
	
public:
	Block(Block* parent=0);
	~Block();
	
	void updateWorldTm();
	
	int numChildBlocks() const;
	
	Matrix44F* tmR();
	const Matrix44F& worldTm() const;

protected:
	void addChild(Block* child);
	
};

}

class BlockDeformerBuilder {

	std::vector<deform::Block* > m_blocks;
	float m_ySegment;

public:
	BlockDeformerBuilder();
	virtual ~BlockDeformerBuilder();
/// local pnt and block ind
	virtual void bindVertexToBlock(Vector3F& pnt, int& iblock) const;
	
	void addBlock(deform::Block* b);
	int numBlocks() const;
	
	deform::Block* getBlock(int i) const;
	
	void setYSeg(float x);
	
protected:

private:
};

class BlockDeformer : public TriangleMeshDeformer {

typedef deform::Block* BlockPtrType;

	BlockPtrType* m_blocks;
	int m_numBlocks;	
/// bend-x, twist-y, roll-z rotation, scale_xz, scale_y
	float m_angles[5];
/// 16 per vertex, 12 for local p, 4 for block ind
	boost::scoped_array<char> m_bind;
		
public:
    BlockDeformer();
	virtual ~BlockDeformer();
	
	void createBlockDeformer(const ATriangleMesh* mesh,
				const BlockDeformerBuilder& builder);
	
	void setBend(const float& x);
	void setTwist(const float& x);
	void setRoll(const float& x);
/// x[0] is y scale x[1] is x and z scale
	void setScaling(const float* x);
	
	virtual void deform(const ATriangleMesh * mesh);
	
protected:
	const float& bendAngle() const;
	const float& twistAngle() const;
	const float& rollAngle() const;
	void updateBlocks();
	BlockPtrType getBlock(int i);
/// local pnt and block ind of i-th vertex
	void getBind(Vector3F& plocal, int& iblock, const int&i) const;
	
private:
	
};

}
#endif
