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

bool MonopodialAttribs::isGeomBranchingUnit() const
{ return true; }

gar::BranchingUnitType MonopodialAttribs::getBranchingUnitType() const
{ return gar::buMonopodial; }

bool MonopodialAttribs::selectBud(gar::SelectBudContext* ctx) const
{
	if(ctx->_budType == gar::bdTerminal)
		return selectTerminalBud(ctx);
	
	return selectLateralBud(ctx);
}

/// root to tip
/// even is left odd is right
/// last is terminal
static const float sBubTm[5][16] = {
{0.510554, 0.428406, 0.604816, 0, -0.578509, 0.68944, 0, 0, -0.463316, -0.388769, 0.666481, 0, -0.224361, 2.258785, 0, 1, },
{-0.385701, 0.323641, -0.745983, 0, 0.578509, 0.68944, 0, 0, 0.571456, -0.479509, -0.503496, 0, 0.293216, 2.186327, 0, 1, },
{-0.385429, -0.323413, 0.746223, 0, -0.578509, 0.68944, 0, 0, -0.57164, -0.479663, -0.503142, 0, -0.224361, 2.198655, 0, 1, },
{0.424016, -0.355792, -0.709664, 0, 0.578509, 0.68944, 0, 0, 0.543634, -0.456163, 0.553514, 0, 0.293216, 2.156501, 0, 1,},
{-0.5, 0, 0.866025, 0, 0, 1, 0, 0, -0.866025, 0, -0.5, 0, 0, 2.5, 0, 1},
};

static const int sBubBind[5] = {
0, 2, 4, 6, 7
};

static const int sLateralBudPermeate[8][4] = {
{0, 3, 2, 1},
{0, 1, 2, 3},
{2, 3, 0, 1},
{2, 1, 0, 3},
{1, 0, 3, 2},
{1, 2, 3, 0},
{3, 0, 1, 2},
{3, 2, 1, 0},
};

bool MonopodialAttribs::selectTerminalBud(gar::SelectBudContext* ctx) const
{		
	float* tm = ctx->_budTm[0];
	memcpy(tm, sBubTm[4], 64);
	
	ctx->_budBind[0] = sBubBind[4];
	
	return true;
}

bool MonopodialAttribs::selectLateralBud(gar::SelectBudContext* ctx) const
{		
	ctx->_numSelect = 4;
	
	const int* perm = sLateralBudPermeate[rand() & 7];
	for(int i=0;i<4;++i) {
		float* tm = ctx->_budTm[i];
		memcpy(tm, sBubTm[perm[i] ], 64);
		ctx->_budBind[i] = sBubBind[perm[i] ];
	}
	
	return true;
}
