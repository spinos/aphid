/*
 *  MonopodialAttribs.cpp
 *  
 *
 *  Created by jian zhang on 8/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "MonopodialAttribs.h"
#include <geom/ATriangleMesh.h>
#include <gar_common.h>
#include <data/monopodial.h>

using namespace aphid;

MonopodialAttribs::MonopodialAttribs() : PieceAttrib(gar::gtSplineCylinder)
{
	const int np = sMonopodialNumVertices;
	const int nt = sMonopodialNumTriangleIndices / 3;
	const int * triind = sMonopodialMeshTriangleIndices;
	const float * vertpos = sMonopodialMeshVertices;
	const float * vertnml = sMonopodialMeshNormals;
	const float * vertcol = sMonopodialMeshVertexColors;
	const float * tritexcoord = sMonopodialMeshTriangleTexcoords;
	
	m_cylinder = ATriangleMesh::CreateFromData(np, nt, triind, 
				vertpos, vertnml, vertcol,
				tritexcoord);
}

bool MonopodialAttribs::hasGeom() const
{ return true; }
	
int MonopodialAttribs::numGeomVariations() const
{ return 1; }

ATriangleMesh* MonopodialAttribs::selectGeom(gar::SelectProfile* prof) const
{
	prof->_exclR = 1.57f;
	prof->_height = 20.f;
	return m_cylinder; 
}

bool MonopodialAttribs::update()
{ return true; }

bool MonopodialAttribs::isGeomStem() const
{ return true; }
