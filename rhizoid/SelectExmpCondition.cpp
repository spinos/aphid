/*
 *  SelectExmpCondition.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 5/19/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "SelectExmpCondition.h"

namespace aphid {

SelectExmpCondition::SelectExmpCondition()
{}

SelectExmpCondition::~SelectExmpCondition()
{}

void SelectExmpCondition::setSurfaceNormal(const Vector3F & nml )
{ m_surfNml = nml; }

void SelectExmpCondition::setTransform(const Matrix44F & tm)
{ m_tm = tm; }

const Matrix44F& SelectExmpCondition::transform() const
{ return m_tm; }

const Vector3F& SelectExmpCondition::surfaceNormal() const
{ return m_surfNml; }

}