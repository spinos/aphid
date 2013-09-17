/*
 *  MlTessellate.cpp
 *  mallard
 *
 *  Created by jian zhang on 9/18/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "MlTessellate.h"
#include <MeshTopology.h>
#include <AccPatchMesh.h>
#include <MlFeather.h>
MlTessellate::MlTessellate() {}

void MlTessellate::setBodyMesh(AccPatchMesh * mesh, MeshTopology * topo)
{
	m_body = mesh;
	m_topo = topo;
}

void MlTessellate::setFeather(MlFeather * feather)
{
	m_feather = feather;
}