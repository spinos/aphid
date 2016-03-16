/*
 *  WorldRenderInterface.h
 *  julia
 *
 *  Created by jian zhang on 3/16/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef WORLDRENDERINTERFACE_H
#define WORLDRENDERINTERFACE_H

#include "cu/AllBase.h"

namespace wldr {
    
void setRenderRect(int * src);
void setFrustum(float * src);
    
void render(uint * color,
                float * nearDepth,
                float * farDepth,
				void * branches,
				void * leaves,
				void * ropes,
                int blockx,
                int gridx, int gridy);

void setBoxFaces();

}

#endif        //  #ifndef WORLDRENDERINTERFACE_H
