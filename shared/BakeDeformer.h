/*
 *  BakeDeformer.h
 *  mallard
 *
 *  Created by jian zhang on 10/7/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <BaseFile.h>
#include <BaseDeformer.h>

class BakeDeformer : public BaseFile, public BaseDeformer {
public:
	BakeDeformer();
	virtual ~BakeDeformer();
	
	virtual char load(const char * filename);
private:

};