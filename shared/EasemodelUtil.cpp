/*
 *  EasemodelUtil.cpp
 *  hc
 *
 *  Created by jian zhang on 4/10/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "EasemodelUtil.h"
namespace ESMUtil
{
void copy(EasyModel * esm, BaseMesh * dst)
{
	const unsigned nf = esm->getNumFace();
    
    int *faceCount = esm->getFaceCount();
    int *faceConnection = esm->getFaceConnection();

    dst->_numFaces = 0;
    
    unsigned i, j;
    for(i = 0; i < nf; i++)
        dst->_numFaces += faceCount[i] - 2;
    
    dst->_numFaceVertices = dst->_numFaces * 3;
    
    dst->_indices = new unsigned[dst->_numFaceVertices];
    
    unsigned curTri = 0;
    unsigned curFace = 0;
    for(i = 0; i < nf; i++) {
        for(j = 1; j < faceCount[i] - 1; j++) {
            dst->_indices[curTri] = faceConnection[curFace];
            dst->_indices[curTri + 1] = faceConnection[curFace + j];
            dst->_indices[curTri + 2] = faceConnection[curFace + j + 1];
            curTri += 3;
        }
        curFace += faceCount[i];
    }
    
    float* cvs = esm->getVertexPosition();
    dst->_numVertices = esm->getNumVertex();
    dst->_vertices = new Vector3F[dst->_numVertices];
    
    for(i = 0; i < dst->_numVertices; i++) {
        dst->_vertices[i].x = cvs[i * 3];
        dst->_vertices[i].y = cvs[i * 3 + 1];
        dst->_vertices[i].z = cvs[i * 3 + 2];
    }
}
}