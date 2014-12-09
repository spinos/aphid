/*
 *  hemisphere_implement.h
 *  
 *
 *  Created by jian zhang on 10/1/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef HEMISPHERE_IMPLEMENT_H
#define HEMISPHERE_IMPLEMENT_H
#include <cutil_inline.h>
#include <cuda_runtime_api.h>

extern "C" void hemisphere(float3 *pos, unsigned numVertices);
#endif        //  #ifndef HEMISPHERE_IMPLEMENT_H