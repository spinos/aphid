/*
 *  SplitEvent.h
 *  kdtree
 *
 *  Created by jian zhang on 10/22/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <ClassificationStorage.h>
#include <PartitionBound.h>
#include <IndexArray.h>
#include <PrimitiveArray.h>

class BuildKdTreeContext;

class SplitEvent {
public:
	typedef Primitive* PrimitivePtr;
	SplitEvent();
	
	void setPos(float val);
	void setAxis(int val);
	
	float getPos() const;
	int getAxis() const;
	
	const float getCost() const;
	
	int leftCount() const;
	int rightCount() const;
	
	void calculateTightBBoxes(const BoundingBox &box, BoundingBox &leftBBox, BoundingBox &rightBBox);
	void calculateCost();
	int side(const BoundingBox &box) const;
	
	void verbose() const;

	static int Dimension;
	static unsigned NumPrimitive;
	static unsigned *PrimitiveIndices;
	static BoundingBox *PrimitiveBoxes;
	static BoundingBox ParentBox;
	
private:
	float m_pos;
	int m_axis;
	float m_cost;
	int m_leftTouch;
	int m_rightTouch;
};
