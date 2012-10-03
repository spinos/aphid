/*
 *  BRDFProgram.cpp
 *  
 *
 *  Created by jian zhang on 10/2/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "BRDFProgram.h"

Vector3F BRDFProgram::V = Vector3F(0.f, 0.7071f, 0.7071f);
Vector3F BRDFProgram::N = Vector3F(0.f, 1.f, 0.f);
Vector3F BRDFProgram::Tangent = Vector3F(1.f, 0.f, 0.f);
Vector3F BRDFProgram::Binormal = Vector3F(0.f, 0.f, 1.f);

BRDFProgram::BRDFProgram() {}

void BRDFProgram::setVTheta(float val)
{
    V.y = sin(val);
    V.z = cos(val);
}
