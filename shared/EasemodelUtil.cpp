/*
 *  EasemodelUtil.cpp
 *  hc
 *
 *  Created by jian zhang on 4/10/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#include <EasyModelIn.h>
#include <EasyModelOut.h>
#include "EasemodelUtil.h"

namespace ESMUtil
{
void Import(const char * filename, BaseMesh * dst)
{
	EasyModelIn * esm = new EasyModelIn(filename);
	
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
    
    printf("create pc %i ", nf);
    dst->createPolygonCounts(nf);
    
    unsigned nfv = 0;
    for(i = 0; i < nf; i++) {
        dst->m_polygonCounts[i] = faceCount[i];
        nfv += faceCount[i];
    }
    
    dst->createPolygonIndices(nfv);
    
    nfv = 0;
    for(i = 0; i < nf; i++) {
        for(j = 0; j < faceCount[i]; j++) {
            dst->m_polygonIndices[nfv] = faceConnection[nfv];
            nfv++;
        }
    }
    
    float* cvs = esm->getVertexPosition();
    dst->createVertices(esm->getNumVertex());
    
    for(i = 0; i < dst->_numVertices; i++) {
        dst->_vertices[i].x = cvs[i * 3];
        dst->_vertices[i].y = cvs[i * 3 + 1];
        dst->_vertices[i].z = cvs[i * 3 + 2];
    }
	
	unsigned numQuads = 0;
    for(i = 0; i < nf; i++) {
		if(faceCount[i] == 4)
			numQuads++;
	}
		
	if(numQuads < 1) return;
	
	dst->createQuadIndices(numQuads * 4);
	
	unsigned ie = 0;
	curFace = 0;
	for(i = 0; i < nf; i++) {
		if(faceCount[i] == 4) {
			for(j = 0; j < faceCount[i]; j++) {
				dst->m_quadIndices[ie] = faceConnection[curFace + j];
				ie++;
			}
		}
		curFace += faceCount[i];
	}
	
	delete esm;
}

void Export(const char * filename, BaseMesh * src)
{
	EasyModelOut * esm = new EasyModelOut(filename);
	
	esm->begin();
	
	unsigned numPolygons = src->m_numPolygons;
	int * faceCounts = new int[numPolygons];
	for(unsigned i = 0; i < numPolygons; i++) faceCounts[i] = src->m_polygonCounts[i];
	
	esm->writeFaceCount(numPolygons, faceCounts);
	
	unsigned numFaceVertices = src->m_numPolygonVertices;
	
	int * faceVertices = new int[numFaceVertices];
	for(unsigned i = 0; i < numFaceVertices; i++) faceVertices[i] = src->m_polygonIndices[i];
	
	esm->writeFaceConnection(numFaceVertices, faceVertices);
	
	unsigned numVertices = src->getNumVertices();
	
	Vector3F * vertexPositions = new Vector3F[numVertices];
	for(unsigned i = 0; i < numVertices; i++) vertexPositions[i] = src->getVertices()[i];
	
	esm->writeP(numVertices, vertexPositions);
	esm->end();
	esm->flush();
	
	delete esm;
}
}
