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
#include <map>
namespace ESMUtil
{
void Import(const char * filename, BaseMesh * dst)
{
	EasyModelIn * esm = new EasyModelIn(filename);
	
	baseImport(esm, dst);	
	
	delete esm;
	
	dst->processTriangleFromPolygon();
	dst->processQuadFromPolygon();
}

void ImportPatch(const char * filename, PatchMesh * dst)
{
	EasyModelIn * esm = new EasyModelIn(filename);
	
	baseImport(esm, dst);	
	
	dst->createPolygonUV(esm->getNumUVs(), esm->getNumUVIds());
	
	for(unsigned i = 0; i < esm->getNumUVs(); i++) {
		dst->us()[i] = esm->getUs()[i];
		dst->vs()[i] = esm->getVs()[i];
	}
	
	for(unsigned i = 0; i < esm->getNumUVIds(); i++)
		dst->uvIds()[i] = esm->getUVIds()[i];
	
	dst->processTriangleFromPolygon();
	dst->processQuadFromPolygon();
	
	delete esm;
}

void Export(const char * filename, BaseMesh * src)
{
	EasyModelOut * esm = new EasyModelOut(filename);
	
	esm->begin();
	
	unsigned numPolygons = src->getNumPolygons();
	int * faceCounts = new int[numPolygons];
	for(unsigned i = 0; i < numPolygons; i++) faceCounts[i] = src->polygonCounts()[i];
	
	esm->writeFaceCount(numPolygons, faceCounts);
	
	unsigned numFaceVertices = src->getNumPolygonFaceVertices();
	
	int * faceVertices = new int[numFaceVertices];
	for(unsigned i = 0; i < numFaceVertices; i++) faceVertices[i] = src->polygonIndices()[i];
	
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
	dst->cleanup();
	
	const unsigned nf = esm->getNumFace();
    
    int *faceCount = esm->getFaceCount();
    int *faceConnection = esm->getFaceConnection();

	unsigned i, j;
	
	std::map<unsigned, unsigned> faceCountList;
	for(i = 0; i < nf; i++) {
		if(faceCountList.find(faceCount[i]) == faceCountList.end()) faceCountList[faceCount[i]] = 1;
		else faceCountList[faceCount[i]] += 1;
	}
	
	std::map<unsigned, unsigned>::const_iterator it = faceCountList.begin();
	for(; it != faceCountList.end(); ++it) {
		std::clog<<" num "<<(*it).first<<" sided faces: "<<(*it).second<<"\n";
	}
    
    dst->createPolygonCounts(nf);
    
    unsigned nfv = 0;
    for(i = 0; i < nf; i++) {
        dst->polygonCounts()[i] = faceCount[i];
        nfv += faceCount[i];
    }
    
    dst->createPolygonIndices(nfv);
    const unsigned nfacev = nfv;
    nfv = 0;
    for(i = 0; i < nf; i++) {
        for(j = 0; j < faceCount[i]; j++) {
            dst->polygonIndices()[nfv] = faceConnection[nfv];
            nfv++;
        }
    }
	
	if(nfv == nfacev) std::cout<<"\nn face v equals\n";
    
    float* cvs = esm->getVertexPosition();
    dst->createVertices(esm->getNumVertex());
    
    for(i = 0; i < dst->getNumVertices(); i++)
        dst->vertices()[i].set(cvs[i * 3], cvs[i * 3 + 1], cvs[i * 3 + 2]);
}
}
