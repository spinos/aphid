/*
 *  EasyModelOut.cpp
 *  easymodel
 *
 *  Created by jian zhang on 4/25/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "EasyModelOut.h"

EasyModelOut::EasyModelOut(const char* filename) 
{
	_doc.create(filename);
	_doc.openBinFile(filename);
	_doc.recordTime();
	m_fileName = filename;
}

EasyModelOut::~EasyModelOut() {}

void EasyModelOut::begin()
{
	_doc.animBegin("|mesh", AnimIO::tTransform);
	_doc.animEnd();
	_doc.animBegin("|mesh|meshShape", AnimIO::tMesh);
}

void EasyModelOut::end()
{
	_doc.animEnd();
}

void EasyModelOut::writeFaceCount(unsigned numPolygon, int * faceCounts)
{
	_doc.addFaceCount(numPolygon, faceCounts);
}

void EasyModelOut::writeFaceConnection(unsigned numFaceVertices, int * faceVertices)
{
	_doc.addFaceConnection(numFaceVertices, faceVertices);
}
	
void EasyModelOut::writeP(unsigned numVertices, Vector3F * vertexPositions)
{
	_doc.addP(numVertices, vertexPositions);
}

void EasyModelOut::flush()
{
	_doc.recordDataSize();
	_doc.closeBinFile();
	_doc.save(m_fileName.c_str());
	_doc.free();
}