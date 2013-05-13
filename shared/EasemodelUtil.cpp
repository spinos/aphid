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
	
	baseImport(esm, dst);	
	
	delete esm;
	
	dst->processTriangleFromPolygon();
	dst->processQuadFromPolygon();
	dst->processRealEdgeFromPolygon();
}

void ImportPatch(const char * filename, PatchMesh * dst)
{
	EasyModelIn * esm = new EasyModelIn(filename);
	
	baseImport(esm, dst);	
	
	dst->processTriangleFromPolygon();
	dst->processQuadFromPolygon();
	dst->processRealEdgeFromPolygon();
	
	dst->prePatchValence();
	
	for(int i = 0; i < dst->numPatches(); i++) {
		for(unsigned j = 0; j < 24; j++)
			dst->patchVertices()[i*24 + j] = esm->getPatchVertex()[i*24 + j];
		for(unsigned j = 0; j < 15; j++)
			dst->patchBoundaries()[i*15 + j] = esm->getPatchBoundary()[i*15 + j];
	}
	
	for(int i = 0; i < dst->getNumVertices(); i++)
		dst->vertexValence()[i] = esm->getVertexValence()[i];
	
	dst->prePatchUV(esm->getNumUVs(), esm->getNumUVIds());
	
	for(int i = 0; i < esm->getNumUVs(); i++) {
		dst->us()[i] = esm->getUs()[i];
		dst->vs()[i] = esm->getVs()[i];
	}
	
	for(int i = 0; i < esm->getNumUVIds(); i++)
		dst->uvIds()[i] = esm->getUVIds()[i];
	
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

void baseImport(EasyModelIn *esm, BaseMesh * dst)
{
	const unsigned nf = esm->getNumFace();
    
    int *faceCount = esm->getFaceCount();
    int *faceConnection = esm->getFaceConnection();

	unsigned i, j;
    
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
	
	float * nors = esm->getVertexNormal();
	for(i = 0; i < dst->_numVertices; i++) {
        dst->normals()[i].x = nors[i * 3];
        dst->normals()[i].y = nors[i * 3 + 1];
        dst->normals()[i].z = nors[i * 3 + 2];
    }

}
}
