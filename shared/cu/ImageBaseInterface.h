#ifndef IMAGEBASEINTERFACE_H
#define IMAGEBASEINTERFACE_H

#include "AllBase.h"

namespace imagebase {

void resetImage(uint * color,
                float * depth,
                int blockx,
                uint n);
				
void resetImage(uint * color,
                float * nearDepth,
                float * farDepth,
                int blockx,
                uint n);

}
#endif        //  #ifndef IMAGEBASEINTERFACE_H

