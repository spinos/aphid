/*
 *  HaircapProp.cpp
 *  
 *
 *  Created by jian zhang on 8/10/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "HaircapProp.h"
#include <gar_common.h>
#include <data/grass.h>
#include <data/haircap.h>
#include <geom/ATriangleMesh.h>

using namespace aphid;

ATriangleMesh * HaircapProp::sMeshes[16];
float HaircapProp::sExclRs[16];
bool HaircapProp::sMeshesLoaded = false;
float HaircapProp::sMeanExclR = 1.f;

HaircapProp::HaircapProp()
{
	if(!sMeshesLoaded)
		loadMeshes();
}

void HaircapProp::loadMeshes()
{
	const int gt = gar::ToGrassType(gar::gtClover );
	const int ngeom = gar::GrassGeomDeviations[gt];
	int np = sHaircapNumVertices;
	int nt = sHaircapNumTriangleIndices / 3;
	const int * triind = sHaircapMeshTriangleIndices;
	
	sMeanExclR = 0.f;
	for(int i=0;i<ngeom;++i) {
	
		const float * vertpos = sHaircapMeshVertices[i];
		const float * vertnml = sHaircapMeshNormals[i];
		const float * vertcol = sHaircapMeshVertexColors[i];
		const float * tritexcoord = sHaircapMeshTriangleTexcoords[i];
		const float exclR = sHaircapExclRadius[i];
	
		sMeshes[i] = ATriangleMesh::CreateFromData(np, nt, triind, 
				vertpos, vertnml, vertcol,
				tritexcoord);
				
		sExclRs[i] = exclR;
		sMeanExclR += exclR;
	}
	sMeanExclR /= (float)ngeom;
	
	sMeshesLoaded = true;
}

bool HaircapProp::hasGeom() const
{ return true; }
	
int HaircapProp::numGeomVariations() const
{
	const int gt = gar::ToGrassType(gar::gtClover );
	return gar::GrassGeomDeviations[gt];
}

ATriangleMesh* HaircapProp::selectGeom(gar::SelectProfile* prof) const
{ 
	if(prof->_condition != gar::slIndex)
		prof->_index = rand() % numGeomVariations();
		
	prof->_exclR = sExclRs[prof->_index];
	return sMeshes[prof->_index]; 
}

void HaircapProp::estimateExclusionRadius(float& minRadius)
{
	if(minRadius > sMeanExclR)
		minRadius = sMeanExclR;
}
