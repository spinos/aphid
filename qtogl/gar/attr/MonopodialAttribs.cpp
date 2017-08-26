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
#include <math/miscfuncs.h>

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
	if(ctx->_budType == gar::bdTerminalFoliage)
		return selectTerminalFoliage(ctx);
		
	if(ctx->_budType == gar::bdLateralFoliage)
		return selectLateralFoliage(ctx);

	if(ctx->_budType == gar::bdTerminal)
		return selectTerminalBud(ctx);
	
	return selectLateralBud(ctx);
}

/// root to tip
/// even is left odd is right
/// last is terminal
static const float sBubTm[5][16] = {
{0.567282, 0.476006, 0.672018, 0, -0.642788, 0.766044, 0, 0, -0.514796, -0.431965, 0.740535, 0, -0.224361, 2.258785, 0, 1,},
{-0.428556, 0.359601, -0.828871, 0, 0.642788, 0.766044, 0, 0, 0.634952, -0.532788, -0.55944, 0, 0.224216, 2.186327, 0, 1,},
{-0.428254, -0.359348, 0.829136, 0, -0.642788, 0.766044, 0, 0, -0.635155, -0.532959, -0.559046, 0, -0.224361, 2.098655, 0, 1, },
{0.471129, -0.395324, -0.788515, 0, 0.642788, 0.766044, 0, 0, 0.604038, -0.506848, 0.615015, 0, 0.224216, 2.056501, 0, 1,},
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

static const float sLateralRotAxis[4][3] = {
{0,0,-1},
{0,0,1},
{0,0,-1},
{0,0,1},
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
		
		Quaternion qz(ctx->_ascending + ctx->_ascendVaring * (.5f + .5f * RandomF01() ), Vector3F(sLateralRotAxis[perm[i] ] ) );
		Matrix33F rotz(qz);
		
		Matrix33F roty;
		roty.rotateY(RandomFn11() * .2f);
		
		Matrix44F mat(sBubTm[perm[i] ]);
		mat *= rotz * roty;
		mat.glMatrix(tm);

		ctx->_budBind[i] = sBubBind[perm[i] ];
	}
	
	return true;
}

bool MonopodialAttribs::selectTerminalFoliage(gar::SelectBudContext* ctx) const
{
	Matrix33F zrot;
	zrot.rotateZ(RandomFn11() * .1f);
	
	Matrix33F yrot;
	yrot.rotateY(RandomFn11() * .5f);
	
	Matrix33F xrot;
	xrot.rotateX(RandomF01() * -.1f);
	
	Matrix44F mat;
	mat.setTranslation(0.f, 2.5f, 0.f);
	
	mat *= xrot * yrot * zrot;
	
	float* tm = ctx->_budTm[0];
	mat.glMatrix(tm);
	
	ctx->_budBind[0] = sBubBind[4];
	
	return true;
}
	
static const float sFoliageTm[4][16] = {
{0.f, 0.f, -1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f,-0.224361f, 2.258785f, 0.f, 1.f,},
{0.f, 0.f,  1.f, 0.f, 0.f, 1.f, 0.f, 0.f,-1.f, 0.f, 0.f, 0.f, 0.224216f, 2.186327f, 0.f, 1.f,},
{0.f, 0.f, -1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f,-0.224361f, 2.098655f, 0.f, 1.f,},
{0.f, 0.f,  1.f, 0.f, 0.f, 1.f, 0.f, 0.f,-1.f, 0.f, 0.f, 0.f, 0.224216f, 2.056501f, 0.f, 1.f,},
};

static const float sFoliageRotAxis[4][3] = {
{0,0,1},
{0,0,-1},
{0,0,1},
{0,0,-1},
};

bool MonopodialAttribs::selectLateralFoliage(gar::SelectBudContext* ctx) const
{
	ctx->_numSelect = 4;
	
	for(int i=0;i<4;++i) {
		float* tm = ctx->_budTm[i];
		
		Quaternion qaxil(ctx->_axil, Vector3F(sFoliageRotAxis[i ] ) );
		Matrix33F rotaxil(qaxil);
		
		Matrix44F mat(sFoliageTm[i ]);
		mat *= rotaxil;
		mat.glMatrix(tm);

		ctx->_budBind[i] = sBubBind[i];
	}
	
	return true;
}
