/*
 *  HemisphereProgram.cpp
 *  
 *
 *  Created by jian zhang on 10/1/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "HemisphereProgram.h"

#include "hemisphere_implement.h"

HemisphereProgram::HemisphereProgram() {}
HemisphereProgram::~HemisphereProgram() {}

void HemisphereProgram::run(CUDABuffer * buffer, BaseMesh * mesh)
{
	float3 *dptr;
	map(buffer, (void **)&dptr);
	
	hemisphere(dptr, mesh->getNumVertices());

	unmap(buffer);
}