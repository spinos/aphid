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
	
	void prePatchUV(unsigned numUVs, unsigned numUVIds);
	
	unsigned numPatches() const;
	
	float * us();
	float * vs();
	unsigned * uvIds();
private:
	unsigned m_numUVs, m_numUVIds;
	float * m_u;
	float * m_v;
	unsigned * m_uvIds;
};