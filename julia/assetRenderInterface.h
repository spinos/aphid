#ifndef ASSET_RENDERINTERFACE_H
#define ASSET_RENDERINTERFACE_H

#include "cu/AllBase.h"

namespace assr {
    
void setRenderRect(int * src);
void setFrustum(float * src);

void drawPyramid(uint * color,
                float * depth,
                int blockx,
                int gridx, int gridy,
				void * planes,
				void * bounding);

}

#endif        //  #ifndef CUBERENDERINTERFACE_H

