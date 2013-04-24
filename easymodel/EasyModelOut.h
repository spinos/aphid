/*
 *  EasyModelOut.h
 *  easymodel
 *
 *  Created by jian zhang on 4/25/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <modelIO.h>
#include <string>

class EasyModelOut {
public:
	EasyModelOut(const char* filename);
	~EasyModelOut();
	
	void begin();
	void end();
	
	void writeFaceCount(unsigned numPolygon, int * faceCounts); 
	void writeFaceConnection(unsigned numFaceVertices, int * faceVertices);
	void writeP(unsigned numVertices, Vector3F * vertexPositions);

	void flush();
private:
	ModelIO _doc;
	std::string m_fileName;
};