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
	~SplitEvent();
	
	void clear();
	
	void setPos(float val);
	void setAxis(int val);
	
	float getPos() const;
	int getAxis() const;
	
	void calculateSides(const PrimitivePtr *primitives, const unsigned &count);
	
	static int Dimension;
	//static BuildKdTreeContext *Context;
	//ClassificationStorage m_sides;
	
private:
	float m_pos;
	int m_axis;
};
