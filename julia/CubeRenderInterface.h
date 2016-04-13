#ifndef CUBERENDERINTERFACE_H
#define CUBERENDERINTERFACE_H

#include "cu/AllBase.h"

namespace cuber {
    
void setRenderRect(int * src);
void setFrustum(float * src);
    
void render(uint * color,
                float * depth,
                int blockx,
                int gridx, int gridy);
				
void drawPyramid(uint * color,
                float * depth,
                int blockx,
                int gridx, int gridy,
				void * planes,
				void * bounding);

void drawVoxel(uint * color,
                float * depth,
                int blockx,
                int gridx, int gridy,
				void * voxels);

void setBoxFaces();

}

#endif        //  #ifndef CUBERENDERINTERFACE_H

