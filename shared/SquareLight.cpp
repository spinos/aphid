/*
 *  SquareLight.cpp
 *  aphid
 *
 *  Created by jian zhang on 1/10/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "SquareLight.h"

SquareLight::SquareLight() 
{
	setEntityType(TSquareLight);
	m_square.set(-1.f, -1.f, 1.f, 1.f);
}

SquareLight::~SquareLight() {}

void SquareLight::setSquare(const BoundingRectangle & square)
{
	m_square = square;
}

BoundingRectangle SquareLight::square() const
{
	return m_square;
}