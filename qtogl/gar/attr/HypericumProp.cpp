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
float HypericumProp::sMeanExclR = 1.f;

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
		
	sMeanExclR = 0.f;
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
		sMeanExclR += exclR;
	}
	sMeanExclR /= (float)ngeom;
	
	sMeshesLoaded = true;
}

bool HypericumProp::hasGeom() const
{ return true; }
	
int HypericumProp::numGeomVariations() const
{
	const int gt = gar::ToGrassType(gar::gtClover );
	return gar::GrassGeomDeviations[gt];
}

ATriangleMesh* HypericumProp::selectGeom(gar::SelectProfile* prof) const
{ 
	if(prof->_condition != gar::slIndex)
		prof->_index = rand() % numGeomVariations();
		
	prof->_exclR = sExclRs[prof->_index];
	return sMeshes[prof->_index]; 
}

void HypericumProp::estimateExclusionRadius(float& minRadius)
{
	if(minRadius > sMeanExclR)
		minRadius = sMeanExclR;
}
