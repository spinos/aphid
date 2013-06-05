/*
 *  Fabric.h
 *  knitfabric
 *
 *  Created by jian zhang on 6/4/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <YarnPatch.h>
class PatchMesh;
class MeshTopology;

class Fabric {
public:
	Fabric();
	
	void setMesh(PatchMesh * mesh, MeshTopology * topo);
	unsigned numPatches() const;
	YarnPatch getPatch(unsigned idx) const;
	YarnPatch *patch(unsigned idx);
private:
	PatchMesh * m_mesh;
	YarnPatch* m_bezier;
};