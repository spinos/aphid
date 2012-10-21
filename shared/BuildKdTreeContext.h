/*
 *  BuildKdTreeContext.h
 *  kdtree
 *
 *  Created by jian zhang on 10/20/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <BoundingBox.h>
#include <PrimitiveArray.h>
#include <IndexArray.h>
#include <KdTreeNodeArray.h>
#include <BaseMesh.h>
#include <SplitCandidate.h>
#include <ClassificationStorage.h>

class PartitionBound {
public:
	PartitionBound() {}
	
	unsigned numPrimitive() {
		return parentMax - parentMin;
	}
	
	unsigned leftCount() {
		return leftChildMax - leftChildMin;
	}
	
	unsigned rightCount() {
		return rightChildMax - rightChildMin;
	}
	
	SplitCandidate bestSplit()
	{
		int axis = bbox.getLongestAxis();
		float pos = bbox.getMin(axis) * 0.5f + bbox.getMax(axis) * 0.5f;
		SplitCandidate candidate;
		candidate.setPos(pos);
		candidate.setAxis(axis);
		return candidate;
	}
	
	BoundingBox bbox;
	unsigned parentMin, parentMax;
	unsigned leftChildMin, leftChildMax;
	unsigned rightChildMin, rightChildMax;
};

class BuildKdTreeContext {
public:
	BuildKdTreeContext();
	~BuildKdTreeContext();
	void appendMesh(BaseMesh* mesh);	
	void initIndices();
	
	void partition(const SplitCandidate & split, PartitionBound & bound);
	
	const unsigned getNumPrimitives() const;
	
	const BoundingBox calculateTightBBox();
	
	KdTreeNode *createTreeBranch();
	KdTreeNode *firstTreeBranch();

	void verbose() const;
	
private:
	PrimitiveArray m_primitives;
	IndexArray m_indices;
	KdTreeNodeArray m_nodes;
};