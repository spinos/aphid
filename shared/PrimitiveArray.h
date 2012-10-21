/*
 *  PrimitiveArray.h
 *  kdtree
 *
 *  Created by jian zhang on 10/20/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <BaseArray.h>
#include <Primitive.h>
#include <Triangle.h>

class PrimitiveArray : public BaseArray {
public:
	PrimitiveArray();
	virtual ~PrimitiveArray();
	
	Primitive * asPrimitive(unsigned index);
	Primitive * asPrimitive();
};