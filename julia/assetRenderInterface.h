#ifndef ASSET_RENDERINTERFACE_H
#define ASSET_RENDERINTERFACE_H

#include "cu/AllBase.h"

namespace assr {
    
void setRenderRect(int * src);
void setFrustum(float * src);
void drawCube(uint * color,
                float * nearDepth,
                float * farDepth,
				int blockx,
                int gridx, int gridy,
                void * branches,
				void * leaves,
				void * ropes,
				int * indirections,
				void * primitives
                );

}

#endif        //  #ifndef CUBERENDERINTERFACE_H

