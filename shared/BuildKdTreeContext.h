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
#include <BoundingBoxList.h>
#include <IndexList.h>
#include <SplitEvent.h>
#include <BuildKdTreeStream.h>

class BuildKdTreeContext {
public:
	BuildKdTreeContext();
	BuildKdTreeContext(BuildKdTreeStream &data);
	~BuildKdTreeContext();
	
	void create(const unsigned &count);

	const unsigned getNumPrimitives() const;

	unsigned *indices();
	
	void verbose() const;
	
	BoundingBoxList m_primitiveBoxes;
	
	void setBBox(const BoundingBox &bbox);
	BoundingBox getBBox() const;
	
private:
	BoundingBox m_bbox;
	IndexList m_indices;
	unsigned m_numPrimitive;
};