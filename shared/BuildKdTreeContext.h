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
#include <BaseMesh.h>
#include <SplitCandidate.h>
#include <ClassificationStorage.h>
class BuildKdTreeContext {
public:
	BuildKdTreeContext();
	void appendMesh(BaseMesh* mesh);	
	void initIndices();
	SplitCandidate bestSplit();
	void partition(const SplitCandidate & split);
	
	void setBBox(const BoundingBox &bbox);
	void setPrimitives(const PrimitiveArray &prims);
	void setIndices(const IndexArray &indices);
	
	const unsigned getNumPrimitives() const;
	const BoundingBox & getBBox() const;
	const PrimitiveArray &getPrimitives() const;
	const IndexArray &getLeftIndices() const;
	const IndexArray &getRightIndices() const;
	
	const BoundingBox calculateTightBBox() const;

	void verbose() const;
	
private:
	BoundingBox m_bbox;
	PrimitiveArray m_primitives;
	IndexArray m_indices;
	IndexArray m_leftIndices;
	IndexArray m_rightIndices;
};