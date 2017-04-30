/*
 *  SquareLight.h
 *  aphid
 *
 *  Created by jian zhang on 1/10/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include "BaseLight.h"
#include <BoundingRectangle.h>
class SquareLight : public BaseLight {
public:
	SquareLight();
	virtual ~SquareLight();
	
	virtual const Type type() const;
	
	void setSquare(const BoundingRectangle & square);
	BoundingRectangle square() const;
protected:

private:
	BoundingRectangle m_square;
};