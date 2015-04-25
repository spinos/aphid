/*
 *  DistantLight.h
 *  aphid
 *
 *  Created by jian zhang on 1/10/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include "BaseLight.h"

class DistantLight : public BaseLight {
public:
	DistantLight();
	virtual ~DistantLight();
	
	virtual const Type type() const;
protected:

private:

};