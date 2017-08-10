/*
 *  HypericumProp.cpp
 *  
 *
 *  Created by jian zhang on 8/10/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "HypericumProp.h"
#include <gar_common.h>
#include <data/grass.h>
#include <data/hypericum.h>
#include <geom/ATriangleMesh.h>

using namespace aphid;

ATriangleMesh * HypericumProp::sMeshes[16];
float HypericumProp::sExclRs[16];
bool HypericumProp::sMeshesLoaded = false;

HypericumProp::HypericumProp()
{
	if(!sMeshesLoaded)
		loadMeshes();
}

void HypericumProp::loadMeshes()
{
	const int gt = gar::ToGrassType(gar::gtClover );
	const int ngeom = gar::GrassGeomDeviations[gt];
	int np = sHypericumNumVertices;
	int nt = sHypericumNumTriangleIndices / 3;
	const int * triind = sHypericumMeshTriangleIndices;
		
	for(int i=0;i<ngeom;++i) {
	
		const float * vertpos = sHypericumMeshVertices[i];
		const float * vertnml = sHypericumMeshNormals[i];
		const float * vertcol = sHypericumMeshVertexColors[i];
		const float * tritexcoord = sHypericumMeshTriangleTexcoords[i];
		const float exclR = sHypericumExclRadius[i];
	
		sMeshes[i] = ATriangleMesh::CreateFromData(np, nt, triind, 
				vertpos, vertnml, vertcol,
				tritexcoord);
				
		sExclRs[i] = exclR;
	}
	
	sMeshesLoaded = true;
}

bool HypericumProp::hasGeom() const
{ return true; }
	
int HypericumProp::numGeomVariations() const
{
	const int gt = gar::ToGrassType(gar::gtClover );
	return gar::GrassGeomDeviations[gt];
}

ATriangleMesh* HypericumProp::selectGeom(int x, float& exclR) const
{ 
	exclR = sExclRs[x];
	return sMeshes[x]; 
}

