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
#include <ClassificationStorage.h>
#include <PartitionBound.h>
#include <SplitEvent.h>
#include <BuildKdTreeStream.h>

class BuildKdTreeContext {
public:
	BuildKdTreeContext();
	BuildKdTreeContext(BuildKdTreeStream &data);
	~BuildKdTreeContext();
	
	void create(const unsigned &count);

	const unsigned getNumPrimitives() const;

	void setPrimitiveIndex(const unsigned &idx, const unsigned &val);
	void setPrimitiveBBox(const unsigned &idx, const BoundingBox &val);
	
	const unsigned *getIndices() const;
	unsigned *indices();
	
	void verbose() const;
	
	BoundingBox *m_primitiveBoxes;
	
	void setBBox(const BoundingBox &bbox);
	BoundingBox getBBox() const;
	
private:
	BoundingBox m_bbox;
	unsigned *m_indices;
	unsigned m_numPrimitive;
};