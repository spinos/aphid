/*
 *  bezierPatch.cpp
 *  catmullclark
 *
 *  Created by jian zhang on 10/26/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include <AllMath.h>
#include <BoundingBox.h>
#include "bezierPatch.h"

BezierPatch::BezierPatch() {}
BezierPatch::~BezierPatch() {}

void BezierPatch::setTexcoord(float* u, float* v, unsigned* idx)
{
	_texcoords[0].x = u[idx[0]];
	_texcoords[1].x = u[idx[1]];
	_texcoords[2].x = u[idx[3]];
	_texcoords[3].x = u[idx[2]];
	
	_texcoords[0].y = v[idx[0]];
	_texcoords[1].y = v[idx[1]];
	_texcoords[2].y = v[idx[3]];
	_texcoords[3].y = v[idx[2]];
	
	float smin = _texcoords[0].x;
	if(_texcoords[1].x < smin) smin = _texcoords[1].x;
	if(_texcoords[2].x < smin) smin = _texcoords[2].x;
	if(_texcoords[3].x < smin) smin = _texcoords[3].x;
	
	float smax = _texcoords[0].x;
	if(_texcoords[1].x > smin) smax = _texcoords[1].x;
	if(_texcoords[2].x > smin) smax = _texcoords[2].x;
	if(_texcoords[3].x > smin) smax = _texcoords[3].x;
	
	float tmin = _texcoords[0].y;
	if(_texcoords[1].y < tmin) tmin = _texcoords[1].y;
	if(_texcoords[2].y < tmin) tmin = _texcoords[2].y;
	if(_texcoords[3].y < tmin) tmin = _texcoords[3].y;
	
	float tmax = _texcoords[0].y;
	if(_texcoords[1].y > tmin) tmax = _texcoords[1].y;
	if(_texcoords[2].y > tmin) tmax = _texcoords[2].y;
	if(_texcoords[3].y > tmin) tmax = _texcoords[3].y;
	
	float delta = smax - smin;
	if(tmax - tmin > delta) delta = tmax - tmin;
	//_lodBase = (int)log2f(1.0 / delta) + 1;
}

Vector3F BezierPatch::p(unsigned u, unsigned v) const
{
	return _contorlPoints[4 * v + u];
}

Vector3F BezierPatch::normal(unsigned u, unsigned v) const
{
	return _normals[4 * v + u];
}

Vector2F BezierPatch::tex(unsigned u, unsigned v) const
{
	return _texcoords[2 * v + u];
}

void BezierPatch::evaluateContolPoints()
{
	_contorlPoints[0] = Vector3F(-2.f, -1.1f, 1.5f);
	_contorlPoints[1] = Vector3F(0.f, -1.05f, 1.1f);
	_contorlPoints[2] = Vector3F(1.f, -1.3f, 1.f);
	_contorlPoints[3] = Vector3F(2.f, -2.f, 1.4f);
	
	_contorlPoints[4] = Vector3F(-1.f, .14f, 0.f);
	_contorlPoints[5] = Vector3F(0.f, .26f, 0.2f);
	_contorlPoints[6] = Vector3F(1.4f, -.45f, 0.3f);
	_contorlPoints[7] = Vector3F(2.5f, -1.2f, 0.17f);
	
	_contorlPoints[8] = Vector3F(-1.f, 1.5f, -1.f);
	_contorlPoints[9] = Vector3F(.41f, 1.1f, -1.2f);
	_contorlPoints[10] = Vector3F(1.8f, .3f, -1.6f);
	_contorlPoints[11] = Vector3F(2.9f, -.4f, -1.5f);
	
	_contorlPoints[12] = Vector3F(-1.f, 3.f, -2.f);
	_contorlPoints[13] = Vector3F(0.f, 2.f, -2.3f);
	_contorlPoints[14] = Vector3F(1.f, 3.f, -2.5f);
	_contorlPoints[15] = Vector3F(2.f, 4.f, -2.98f);
}

void BezierPatch::evaluateTangents()
{
	for(unsigned j = 0; j < 4; j++)
	{
		for(unsigned i = 0; i < 3; i++)
		{
			_tangents[j * 3 + i] = p(i + 1, j) - p(i, j);
		}
	}
}

void BezierPatch::evaluateBinormals()
{
	for(unsigned j = 0; j < 3; j++)
	{
		for(unsigned i = 0; i < 4; i++)
		{
			_binormals[j * 4 + i] = p(i, j + 1) - p(i, j);
		}
	}
}

void BezierPatch::evaluateSurfacePosition(float u, float v, Vector3F * pos) const
{
	Vector2F L0(1.f-u,1.f-v);
	Vector2F L1(u,v);

	Vector2F B0 =     (L0 * L0) * L0;
	Vector2F B1 = (L0 * L0) * L1 * 3.f;
	Vector2F B2 = (L1 * L1) * L0 * 3.f;
	Vector2F B3 =     (L1 * L1) * L1;

	*pos = 
		(p(0,0) * B0.x + p(1,0) * B1.x + p(2,0) * B2.x + p(3,0) * B3.x) * B0.y +
		(p(0,1) * B0.x + p(1,1) * B1.x + p(2,1) * B2.x + p(3,1) * B3.x) * B1.y +
		(p(0,2) * B0.x + p(1,2) * B1.x + p(2,2) * B2.x + p(3,2) * B3.x) * B2.y +
		(p(0,3) * B0.x + p(1,3) * B1.x + p(2,3) * B2.x + p(3,3) * B3.x) * B3.y;
}

void BezierPatch::evaluateSurfaceTangent(float u, float v, Vector3F * tang) const
{
	Vector2F L0(1-u,1-v);
	Vector2F L1(u,v);
	
	Vector2F Q0 =     L0 * L0;
	Vector2F Q1 =  L0 * L1 * 2.f;
	Vector2F Q2 =     L1 * L1;

	Vector2F B0 =     L0 * Q0;
	Vector2F B1 = Q0 * L1 * 3.f;
	Vector2F B2 = Q2 * L0 * 3.f;
	Vector2F B3 =     Q2 * L1;
	
	*tang = 
		(_tangents[0 ] * B0.y + _tangents[1 ] * B1.y + _tangents[2 ] * B2.y + _tangents[3 ] * B3.y) * Q0.x +
		(_tangents[4 ] * B0.y + _tangents[5 ] * B1.y + _tangents[6 ] * B2.y + _tangents[7 ] * B3.y) * Q1.x +
		(_tangents[8 ] * B0.y + _tangents[9 ] * B1.y + _tangents[10] * B2.y + _tangents[11] * B3.y) * Q2.x;
	tang->normalize();
}

void BezierPatch::evaluateSurfaceBinormal(float u, float v, Vector3F * binm) const
{
	Vector2F L0(1-u,1-v);
	Vector2F L1(u,v);
	
	Vector2F Q0 =     L0 * L0;
	Vector2F Q1 =  L0 * L1 * 2.f;
	Vector2F Q2 =     L1 * L1;

	Vector2F B0 =     L0 * Q0;
	Vector2F B1 = Q0 * L1 * 3.f;
	Vector2F B2 = Q2 * L0 * 3.f;
	Vector2F B3 =     Q2 * L1;
	
	*binm = 
		(_binormals[0 ] * B0.x + _binormals[1 ] * B1.x + _binormals[2 ] * B2.x + _binormals[3 ] * B3.x) * Q0.y +
		(_binormals[4 ] * B0.x + _binormals[5 ] * B1.x + _binormals[6 ] * B2.x + _binormals[7 ] * B3.x) * Q1.y +
		(_binormals[8 ] * B0.x + _binormals[9 ] * B1.x + _binormals[10] * B2.x + _binormals[11] * B3.x) * Q2.y;
	binm->normalize();
}

void BezierPatch::evaluateSurfaceNormal(float u, float v, Vector3F * nor) const
{
	Vector2F L0(1.f-u,1.f-v);
	Vector2F L1(u,v);

	Vector2F B0 =     (L0 * L0) * L0;
	Vector2F B1 = (L0 * L0) * L1 * 3.f;
	Vector2F B2 = (L1 * L1) * L0 * 3.f;
	Vector2F B3 =     (L1 * L1) * L1;

	*nor = 
		(normal(0,0) * B0.x + normal(1,0) * B1.x + normal(2,0) * B2.x + normal(3,0) * B3.x) * B0.y +
		(normal(0,1) * B0.x + normal(1,1) * B1.x + normal(2,1) * B2.x + normal(3,1) * B3.x) * B1.y +
		(normal(0,2) * B0.x + normal(1,2) * B1.x + normal(2,2) * B2.x + normal(3,2) * B3.x) * B2.y +
		(normal(0,3) * B0.x + normal(1,3) * B1.x + normal(2,3) * B2.x + normal(3,3) * B3.x) * B3.y;
	nor->normalize();
}

void BezierPatch::evaluateSurfaceTexcoord(float u, float v, Vector3F * texcoord) const
{
	Vector2F L0(1-u,1-v);
	Vector2F L1(u,v);

	Vector2F st = (tex(0, 0) * L0.x + tex(1, 0) * L1.x) * L0.y + 
		(tex(0, 1) * L0.x + tex(1, 1) * L1.x) * L1.y;
	*texcoord = Vector3F(st.x, st.y, 0.f);
}

void BezierPatch::evaluateSurfaceVector(float u, float v, Vector3F * src, Vector3F * dst) const
{
	Vector2F L0(1-u,1-v);
	Vector2F L1(u,v);

	*dst = (src[0] * L0.x + src[1] * L1.x) * L0.y + 
		(src[2] * L0.x + src[3] * L1.x) * L1.y;
}

const BoundingBox BezierPatch::controlBBox() const
{
	BoundingBox box;
	for(int i = 0; i < 16; i++) {
		box.updateMin(_contorlPoints[i]);
		box.updateMax(_contorlPoints[i]);
	}
		
	return box;
}

void BezierPatch::decasteljauSplit(BezierPatch *dst) const
{
    Vector3F split[7][7];

// blue	corner
    split[0][0] = p(0, 0);
    split[0][6] = p(3, 0);
	split[2][0] = p(0, 1);
    split[2][6] = p(3, 1);
	split[4][0] = p(0, 2);
    split[4][6] = p(3, 2);
    split[6][6] = p(3, 3);
    split[6][0] = p(0, 3);
    
    unsigned u, v;
	
// u direction

	for(v = 0; v < 7; v+= 2) {
// green
        for(u = 1; u < 7; u += 2) {
            split[v][u] = (p(u/2, v/2) + p(u/2 + 1, v/2)) * .5f;
        }
// yellow
		split[v][2] = (split[v][1] + split[v][3]) * .5f;
		split[v][4] = (split[v][3] + split[v][5]) * .5f;
// red
		split[v][3] = (split[v][2] + split[v][4]) * .5f;
    }

// v direction
	for(u = 0; u < 7; u += 1) {
// green
		for(v = 1; v <7; v+= 2) {
			split[v][u] = (split[v-1][u] + split[v+1][u]) * .5f;
		}
// yellow		
		split[2][u] = (split[1][u] + split[3][u]) * .5f;
		split[4][u] = (split[3][u] + split[5][u]) * .5f;
// red
		split[3][u] = (split[2][u] + split[4][u]) * .5f;
	}
	
    BezierPatch *child0 = &dst[0];
    BezierPatch *child1 = &dst[1];
    BezierPatch *child2 = &dst[2];
    BezierPatch *child3 = &dst[3];
    for(v = 0; v < 4; v++) {
        for(u = 0; u < 4; u++) {
            child0->_contorlPoints[v * 4 + u] = split[v][u];
            child1->_contorlPoints[v * 4 + u] = split[v][u + 3];
            child2->_contorlPoints[v * 4 + u] = split[v + 3][u + 3];
            child3->_contorlPoints[v * 4 + u] = split[v + 3][u];
        }
    }
}

/*
 *  3 ---- c ---- 2
 *	|	   |      |
 *  |  3   |  2   |
 *  d ---- e ---- b
 *	|	   |      |
 *  |  0   |  1   |
 *  0 ---- a ---- 1
 */

void BezierPatch::splitPatchUV(PatchSplitContext ctx, PatchSplitContext * child) const
{
	Vector2F a = ctx.patchUV[0]/ 2.f + ctx.patchUV[1]/2.f;
	Vector2F b = ctx.patchUV[1]/ 2.f + ctx.patchUV[2]/2.f;
	Vector2F c = ctx.patchUV[2]/ 2.f + ctx.patchUV[3]/2.f;
	Vector2F d = ctx.patchUV[3]/ 2.f + ctx.patchUV[0]/2.f;
	Vector2F e = ctx.patchUV[0]/ 4.f + ctx.patchUV[1]/4.f + ctx.patchUV[2]/ 4.f + ctx.patchUV[3]/4.f;
		
	PatchSplitContext *res = &child[0];

	res->patchUV[0] = ctx.patchUV[0];
	res->patchUV[1] = a;
	res->patchUV[2] = e;
	res->patchUV[3] = d;

	res = &child[1];
	res->patchUV[0] = a;
	res->patchUV[1] = ctx.patchUV[1];
	res->patchUV[2] = b;
	res->patchUV[3] = e;

	res = &child[2];
	res->patchUV[0] = e;
	res->patchUV[1] = b;
	res->patchUV[2] = ctx.patchUV[2];
	res->patchUV[3] = c;
	
	res = &child[3];
	res->patchUV[0] = d;
	res->patchUV[1] = e;
	res->patchUV[2] = c;
	res->patchUV[3] = ctx.patchUV[3];
}
 
void BezierPatch::tangentFrame(float u, float v, Matrix33F & frm) const
{
	Vector3F du;
	evaluateSurfaceTangent(u, v, &du);
	du.normalize();
	
    Vector3F side;
	evaluateSurfaceNormal(u, v, &side);
    side.normalize();
    
    Vector3F up = du.cross(side);
    up.normalize();
	
	du = side.cross(up);
	du.normalize();
	
    frm.fill(side, up, du);
}
//:~