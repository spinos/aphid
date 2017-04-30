/*
 *  phong_implement.h
 *  
 *
 *  Created by jian zhang on 10/1/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef WARD_IMPLEMENT_H
#define WARD_IMPLEMENT_H

extern "C" void ward_brdf(float3 *pos, unsigned numVertices, unsigned width, float3 V, float3 N, float3 X, float3 Y, float alpha_x, float alpha_y, bool anisotropic);
#endif        //  #ifndef WARD_IMPLEMENT_H