/*
 *  AccPatchMesh.h
 *  mallard
 *
 *  Created by jian zhang on 8/30/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once

#include <PatchMesh.h>
class AccPatch;
class MeshTopology;
class AccPatchMesh : public PatchMesh {
public:
	AccPatchMesh();
	virtual ~AccPatchMesh();
	
	void setup(MeshTopology * topo);
	
	AccPatch* beziers() const;
	
private:
	AccPatch* m_bezier;
};
