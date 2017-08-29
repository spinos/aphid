/*
 *  CloverProp.cpp
 *  
 *
 *  Created by jian zhang on 8/10/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "CloverProp.h"
#include <gar_common.h>
#include <data/grass.h>
#include <data/clover.h>
#include <geom/ATriangleMesh.h>

using namespace aphid;

ATriangleMesh * CloverProp::sMeshes[16];
float CloverProp::sExclRs[16];
bool CloverProp::sMeshesLoaded = false;
float CloverProp::sMeanExclR = 1.f;

CloverProp::CloverProp()
{
	if(!sMeshesLoaded)
		loadMeshes();
}

void CloverProp::loadMeshes()
{
	const int gt = gar::ToGrassType(gar::gtClover );
	const int ngeom = gar::GrassGeomDeviations[gt];
	int np = sCloverNumVertices;
	int nt = sCloverNumTriangleIndices / 3;
	const int * triind = sCloverMeshTriangleIndices;
		
	sMeanExclR = 0.f;
	for(int i=0;i<ngeom;++i) {
	
		const float * vertpos = sCloverMeshVertices[i];
		const float * vertnml = sCloverMeshNormals[i];
		const float * vertcol = sCloverMeshVertexColors[i];
		const float * tritexcoord = sCloverMeshTriangleTexcoords[i];
		const float exclR = sCloverExclRadius[i];
	
		sMeshes[i] = ATriangleMesh::CreateFromData(np, nt, triind, 
				vertpos, vertnml, vertcol,
				tritexcoord);
				
		sExclRs[i] = exclR;
		sMeanExclR += exclR;
	}
	sMeanExclR /= (float)ngeom;
	
	sMeshesLoaded = true;
}

bool CloverProp::hasGeom() const
{ return true; }
	
int CloverProp::numGeomVariations() const
{
	const int gt = gar::ToGrassType(gar::gtClover );
	return gar::GrassGeomDeviations[gt];
}

ATriangleMesh* CloverProp::selectGeom(gar::SelectProfile* prof) const
{ 
	if(prof->_condition != gar::slIndex)
		prof->_index = rand() % numGeomVariations();
		
	prof->_exclR = sExclRs[prof->_index];
	return sMeshes[prof->_index]; 
}

void CloverProp::estimateExclusionRadius(float& minRadius)
{
	if(minRadius > sMeanExclR)
		minRadius = sMeanExclR;
}
