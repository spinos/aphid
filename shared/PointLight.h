/*
 *  PointLight.h
 *  aphid
 *
 *  Created by jian zhang on 1/10/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include "BaseLight.h"

class PointLight : public BaseLight {
public:
	PointLight();
	virtual ~PointLight();
	
	virtual const Type type() const;
protected:

private:

};