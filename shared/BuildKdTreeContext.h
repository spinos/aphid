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
#include <PartitionBound.h>
#include <SplitEvent.h>

class BuildKdTreeContext {
public:
	BuildKdTreeContext();
	~BuildKdTreeContext();
	void appendMesh(BaseMesh* mesh);	
	void initIndices();
	
	void partition(const SplitEvent &split, PartitionBound & bound, int leftSide);
	
	const unsigned getNumPrimitives() const;
	const PrimitiveArray &getPrimitives() const;
	const IndexArray &getIndices() const;
	
	const BoundingBox calculateTightBBox();
	
	KdTreeNode *createTreeBranch();
	KdTreeNode *firstTreeBranch();
	void releaseIndicesAt(unsigned loc);

	void verbose() const;
	
private:
	PrimitiveArray m_primitives;
	IndexArray m_indices;
	KdTreeNodeArray m_nodes;
};