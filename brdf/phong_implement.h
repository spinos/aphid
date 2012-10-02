/*
 *  phong_implement.h
 *  
 *
 *  Created by jian zhang on 10/1/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef PHONG_IMPLEMENT_H
#define PHONG_IMPLEMENT_H

extern "C" void phong_brdf(float3 *pos, unsigned width, unsigned height, float3 V, float3 N, float exposure, int divideByNdotL);
#endif        //  #ifndef PHONG_IMPLEMENT_H