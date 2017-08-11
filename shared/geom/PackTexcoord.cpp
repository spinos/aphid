/*
 *  PackTexcoord.h
 *
 *  pack multiple mesh texcoord into [0,1] region
 *  each mesh has its texcoord in [0,1]
 *  w/h ration and x or y fills [0,1]
 *
 */
 
#include "PackTexcoord.h"
#include <geom/ATriangleMesh.h>

namespace aphid {

PackTexcoord::PackTexcoord()
{}

bool PackTexcoord::computeTexcoord(ATriangleMesh* msh[],
        const int& nmshs,
        const float& whratio)
{
    
    if(whratio <= 1.f) {
        return packInRows(msh, nmshs, whratio);
    }
    
    return packInColumns(msh, nmshs, whratio);
}

bool PackTexcoord::packInRows(ATriangleMesh* msh[],
        const int& nmshs,
        const float& whratio)
{
/// start with 1 row
        int nRows = 1;
        float scaling = 1.f;
        int nBlksPerRow = nmshs;               
        float rowWidth = whratio * scaling * nBlksPerRow;
        
/// not enough
        while(rowWidth > 1.f ) {
/// double n rows, half the size and n blocks per row
            nRows = nRows<<1;
            scaling *= .5f;
            nBlksPerRow = (nBlksPerRow + 1)>>1;
            rowWidth = whratio * scaling * nBlksPerRow;
        }
#if 0       
        std::cout<<"\n pack in "<<nRows<<" rows"
            <<" "<<nBlksPerRow<<" blocks per row";
#endif    
        const float yStep = scaling;
        const float xStep = yStep * whratio;
        float xLoc = 0.f;
        float yLoc = 0.f;
        int j = 0;
        for(int i=0;i<nmshs;++i) {
            ATriangleMesh* mshi = msh[i];
            shrinkAndMove(mshi, scaling, xLoc, yLoc);
            j++;
            if(j==nBlksPerRow) {
/// next row
                j = 0;
                xLoc = 0.f;
                yLoc += yStep;
            } else {
/// next col
                xLoc += xStep;
            }
            
        }
        return true;
}

bool PackTexcoord::packInColumns(ATriangleMesh* msh[],
        const int& nmshs,
        const float& whratio)
{
        const float hwratio = 1.f / whratio;
        int nCols = 1;
        float scaling = 1.f;
        int nBlksPerCol = nmshs;               
        float colHeight = hwratio * scaling * nBlksPerCol;
        

        while(colHeight > 1.f ) {
            nCols = nCols<<1;
            scaling *= .5f;
            nBlksPerCol = (nBlksPerCol + 1)>>1;
            colHeight = hwratio * scaling * nBlksPerCol;
        }
#if 0        
        std::cout<<"\n pack in "<<nCols<<" columns"
            <<" "<<nBlksPerCol<<" blocks per column";
#endif       
        const float xStep = scaling;
        const float yStep = xStep * hwratio;
        float xLoc = 0.f;
        float yLoc = 0.f;
        int j = 0;
        for(int i=0;i<nmshs;++i) {
            ATriangleMesh* mshi = msh[i];
            shrinkAndMove(mshi, scaling, xLoc, yLoc);
            j++;
            if(j==nBlksPerCol) {
/// next col
                j = 0;
                yLoc = 0.f;
                xLoc += xStep;
            } else {
/// next row
                yLoc += yStep;
            }
        }
        return true;
}

void PackTexcoord::shrinkAndMove(ATriangleMesh* msh,
        const float& shrinking,
        const float& xLoc,
        const float& yLoc)
{
/// n face-varying texcoord vertices
    const int nfv = msh->numTriangles() * 3;
    Vector2F * texcoordV = (Vector2F *)msh->triangleTexcoords();        
    for(int i=0;i<nfv;++i) {
        texcoordV[i] *= shrinking;
        texcoordV[i].x += xLoc;
        texcoordV[i].y += yLoc;
    }
}

}
