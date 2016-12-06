/*
 *  PrimitiveArray.h
 *  kdtree
 *
 *  Created by jian zhang on 10/20/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <foundation/BaseArray.h>
#include <Primitive.h>
namespace aphid {

class PrimitiveArray : public BaseArray<Primitive> {
public:
	PrimitiveArray();
	virtual ~PrimitiveArray();
	
	Primitive * asPrimitive(unsigned index);
	//Primitive * asPrimitive(unsigned index) const;
	Primitive * asPrimitive();
protected:
	virtual unsigned elementSize() const;
};

}