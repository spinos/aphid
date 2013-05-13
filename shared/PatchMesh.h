/*
 *  PatchMesh.h
 *  catmullclark
 *
 *  Created by jian zhang on 5/13/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <BaseMesh.h>

class PatchMesh : public BaseMesh {
public:
	PatchMesh();
	virtual ~PatchMesh();
	
	void createVertexValence(unsigned num);
	void prePatchValence();
	
	unsigned numPatches() const;
	unsigned * vertexValence();
	unsigned * patchVertices();
	char * patchBoundaries();
private:
	unsigned * m_patchVertices;
	unsigned * m_vertexValence;
	char * m_patchBoundary;
};