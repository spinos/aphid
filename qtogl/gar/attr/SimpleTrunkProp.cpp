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
	int np = sCloverNumVertices;
	int nt = sCloverNumTriangleIndices / 3;
	const int * triind = sCloverMeshTriangleIndices;
		
	const float * vertpos = sCloverMeshVertices;
	const float * vertnml = sCloverMeshNormals;
	const float * vertcol = sCloverMeshVertexColors;
	const float * tritexcoord = sCloverMeshTriangleTexcoords;
	const float exclR = sCloverExclRadius;
	
	sMesh = ATriangleMesh::CreateFromData(np, nt, triind, 
				vertpos, vertnml, vertcol,
				tritexcoord);
				
	sExclR = exclR;
	
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
