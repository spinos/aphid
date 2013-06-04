/*
 *  Fabric.h
 *  knitfabric
 *
 *  Created by jian zhang on 6/4/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

class PatchMesh;
class AccPatch;
class MeshTopology;

class Fabric {
public:
	Fabric();
	
	void setMesh(PatchMesh * mesh, MeshTopology * topo);
	unsigned numPatches() const;
	AccPatch & getPatch(unsigned idx) const;
private:
	PatchMesh * m_mesh;
	AccPatch* m_bezier;
};