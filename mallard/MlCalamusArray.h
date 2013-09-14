/*
 *  MlCalamusArray.h
 *  mallard
 *
 *  Created by jian zhang on 9/14/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <BaseArray.h>
#include "MlCalamus.h"

class MlCalamusArray : public BaseArray {
public:
	MlCalamusArray();
	virtual ~MlCalamusArray();
	
	MlCalamus * asCalamus(unsigned index);
	MlCalamus * asCalamus(unsigned index) const;
	MlCalamus * asCalamus();
};