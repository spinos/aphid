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
	void partition(const SplitCandidate & split);
	
	const unsigned getNumPrimitives() const;
	const BoundingBox calculateTightBBox() const;
	
private:
	BoundingBox m_bbox;
	PrimitiveArray m_primitives;
	IndexArray m_indices;
	IndexArray m_leftIndices;
	IndexArray m_rightIndices;
};