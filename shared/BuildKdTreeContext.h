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
#include <BaseMesh.h>
class BuildKdTreeContext {
public:
	BuildKdTreeContext();
	void appendMesh(BaseMesh* mesh);
	
	const unsigned getNumPrimitives() const;
	const BoundingBox getBBox() const;
	
private:
	BoundingBox m_bbox;
	PrimitiveArray m_primitives;
};