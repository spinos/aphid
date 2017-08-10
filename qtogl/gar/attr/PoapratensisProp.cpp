/*
 *  PoapratensisProp.cpp
 *  
 *
 *  Created by jian zhang on 8/10/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "PoapratensisProp.h"
#include <gar_common.h>
#include <data/grass.h>
#include <data/poapratensis.h>
#include <geom/ATriangleMesh.h>

using namespace aphid;

ATriangleMesh * PoapratensisProp::sMeshes[16];
float PoapratensisProp::sExclRs[16];
bool PoapratensisProp::sMeshesLoaded = false;

PoapratensisProp::PoapratensisProp()
{
	if(!sMeshesLoaded)
		loadMeshes();
}

void PoapratensisProp::loadMeshes()
{
	const int gt = gar::ToGrassType(gar::gtClover );
	const int ngeom = gar::GrassGeomDeviations[gt];
	int np = sPoapratensisNumVertices;
	int nt = sPoapratensisNumTriangleIndices / 3;
	const int * triind = sPoapratensisMeshTriangleIndices;
		
	for(int i=0;i<ngeom;++i) {
	
		const float * vertpos = sPoapratensisMeshVertices[i];
		const float * vertnml = sPoapratensisMeshNormals[i];
		const float * vertcol = sPoapratensisMeshVertexColors[i];
		const float * tritexcoord = sPoapratensisMeshTriangleTexcoords[i];
		const float exclR = sPoapratensisExclRadius[i];
	
		sMeshes[i] = ATriangleMesh::CreateFromData(np, nt, triind, 
				vertpos, vertnml, vertcol,
				tritexcoord);
				
		sExclRs[i] = exclR;
	}
	
	sMeshesLoaded = true;
}

bool PoapratensisProp::hasGeom() const
{ return true; }
	
int PoapratensisProp::numGeomVariations() const
{
	const int gt = gar::ToGrassType(gar::gtClover );
	return gar::GrassGeomDeviations[gt];
}

ATriangleMesh* PoapratensisProp::selectGeom(int x, float& exclR) const
{ 
	exclR = sExclRs[x];
	return sMeshes[x]; 
}

