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
class Fabric {
public:
	Fabric();
	
	void setMesh(PatchMesh * mesh);
private:
	PatchMesh * m_mesh;
};