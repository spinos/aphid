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
float PoapratensisProp::sMeanExclR = 1.f;

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
		
	sMeanExclR = 0.f;
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
		sMeanExclR += exclR;
	}
	sMeanExclR /= (float)ngeom;
	
	sMeshesLoaded = true;
}

bool PoapratensisProp::hasGeom() const
{ return true; }
	
int PoapratensisProp::numGeomVariations() const
{
	const int gt = gar::ToGrassType(gar::gtClover );
	return gar::GrassGeomDeviations[gt];
}

ATriangleMesh* PoapratensisProp::selectGeom(gar::SelectProfile* prof) const
{ 
	if(prof->_condition != gar::slIndex)
		prof->_index = rand() % numGeomVariations();
		
	prof->_exclR = sExclRs[prof->_index];
	return sMeshes[prof->_index]; 
}

void PoapratensisProp::estimateExclusionRadius(float& minRadius)
{
	if(minRadius > sMeanExclR)
		minRadius = sMeanExclR;
}
