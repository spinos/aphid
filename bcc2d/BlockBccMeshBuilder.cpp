/*
 *  BlockBccMeshBuilder.cpp
 *  bcc
 *
 *  Created by jian zhang on 8/24/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "BlockBccMeshBuilder.h"
#include <CartesianGrid.h>
BlockBccMeshBuilder::BlockBccMeshBuilder() 
{
    m_verticesPool = new CartesianGrid;
}
BlockBccMeshBuilder::~BlockBccMeshBuilder() 
{
    delete m_verticesPool;
}

void BlockBccMeshBuilder::build(const AOrientedBox & ob, 
				int gx, int gy, int gz)
{
    const Vector3F center = ob.center();
    const float span = ob.extent().x;
    float originSpan[4];
    originSpan[0] = center.x - span * 1.01f;
    originSpan[1] = center.y - span * 1.01f;
    originSpan[2] = center.z - span * 1.01f;
    originSpan[3] = span * 2.02f;
    m_verticesPool->setBounding(originSpan);
    
    const float cellSize = span * 2.f / (float)gx;
    const float cellSizeH = cellSize * .5f;
    
    const Vector3F cellOrigin(center.x - cellSize * (float)gx * .5f,
                        center.y - cellSize * (float)gy * .5f,
                        center.z - cellSize * (float)gz * .5f);
    Vector3F cellCenter;
    Vector3F tetV[4];
    int i, j, k;
    for(k=0;k<gz;k++) {
        cellCenter.z = cellOrigin.z + cellSize * k;
        for(j=0;j<gy;j++) {
            cellCenter.y = cellOrigin.y + cellSize * j;
            for(i=0;i<gx;i++) {
                cellCenter.x = cellOrigin.x + cellSize * i;
                tetV[0] = cellCenter - Vector3F::YAxis * cellSizeH;
                tetV[1] = tetV[0] + Vector3F::YAxis * cellSize;
                tetV[2] = cellCenter - Vector3F::XAxis * cellSizeH
                                    - Vector3F::ZAxis * cellSizeH;
                tetV[3] = tetV[2] + Vector3F::ZAxis * cellSize;
                addTetrahedron(tetV);
                tetV[2] = tetV[3];
                tetV[3] = tetV[2] + Vector3F::XAxis * cellSize;
                addTetrahedron(tetV);
                tetV[2] = tetV[3];
                tetV[3] = tetV[2] - Vector3F::ZAxis * cellSize;
                addTetrahedron(tetV);
                tetV[2] = tetV[3];
                tetV[3] = tetV[2] - Vector3F::XAxis * cellSize;
                addTetrahedron(tetV);
            }
        }
    }
}

void BlockBccMeshBuilder::addTetrahedron(Vector3F * v)
{
    m_verticesPool->addCell(v[0], 8);
}
//:~