/*
 *  Fabric.cpp
 *  knitfabric
 *
 *  Created by jian zhang on 6/4/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "Fabric.h"
#include <PatchMesh.h>
Fabric::Fabric() {}

void Fabric::setMesh(PatchMesh * mesh)
{
	m_mesh = mesh;
	unsigned numPatch = m_mesh->numPatches();
}