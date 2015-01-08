#ifndef PLANE_IMPLEMENT_H
#define PLANE_IMPLEMENT_H

/*
 *  plane_implement.h
 *  
 *
 *  Created by jian zhang on 10/1/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */


#include <cuda_runtime_api.h>
typedef unsigned int uint;
extern "C" void wavePlane(float4 *pos, unsigned numGrids, float gridSize, float alpha);

#endif        //  #ifndef PLANE_IMPLEMENT_H
