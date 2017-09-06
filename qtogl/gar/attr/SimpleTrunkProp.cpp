/*
 *  SimpleTrunkProp.cpp
 *  
 *
 *  Created by jian zhang on 8/10/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "SimpleTrunkProp.h"
#include <gar_common.h>
#include <data/trunk.h>
#include <data/simple_trunk.h>
#include <geom/ATriangleMesh.h>

using namespace aphid;

ATriangleMesh * SimpleTrunkProp::sMesh;
float SimpleTrunkProp::sExclR;
bool SimpleTrunkProp::sMeshLoaded = false;

SimpleTrunkProp::SimpleTrunkProp()
{
	if(!sMeshLoaded)
		loadMesh();
}

void SimpleTrunkProp::loadMesh()
{
	int np = sAlmondNumVertices;
	int nt = sAlmondNumTriangleIndices / 3;
	const int * triind = sAlmondMeshTriangleIndices;
		
	const float * vertpos = sAlmondMeshVertices;
	const float * vertnml = sAlmondMeshNormals;
	const float * vertcol = sAlmondMeshVertexColors;
	const float * tritexcoord = sAlmondMeshTriangleTexcoords;

	sMesh = ATriangleMesh::CreateFromData(np, nt, triind, 
				vertpos, vertnml, vertcol,
				tritexcoord);
				
	sExclR = sAlmondExclRadius;
	
	sMeshLoaded = true;
}

bool SimpleTrunkProp::hasGeom() const
{ return true; }
	
int SimpleTrunkProp::numGeomVariations() const
{ return 1; }

ATriangleMesh* SimpleTrunkProp::selectGeom(gar::SelectProfile* prof) const
{ 
	prof->_exclR = sExclR;
	return sMesh; 
}

void SimpleTrunkProp::estimateExclusionRadius(float& minRadius)
{
	if(minRadius > sExclR)
		minRadius = sExclR;
}
