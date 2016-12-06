/*
 *  IndexArray.h
 *  kdtree
 *
 *  Created by jian zhang on 10/20/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <foundation/BaseArray.h>

namespace aphid {

class IndexArray : public BaseArray<unsigned> {
public:
	IndexArray();
	virtual ~IndexArray();
	
	unsigned *asIndex();
	unsigned *asIndex(unsigned i);

};

}