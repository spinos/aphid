/*
 *  voxTest.cpp
 *  
 *
 *  Created by jian zhang on 4/13/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "voxTest.h"

namespace aphid {

cvx::Triangle VoxTest::createTriangle(const Vector3F & p0,
								const Vector3F & p1,
								const Vector3F & p2,
								const Vector3F & c0,
								const Vector3F & c1,
								const Vector3F & c2)
{
	cvx::Triangle tri;
	tri.resetNC();
	tri.setP(p0,0);
	tri.setP(p1,1);
	tri.setP(p2,2);
	Vector3F nor = tri.calculateNormal();
	tri.setN(nor, 0);
	tri.setN(nor, 1);
	tri.setN(nor, 2);
	tri.setC(c0, 0);
	tri.setC(c1, 1);
	tri.setC(c2, 2);
	return tri;
}

void VoxTest::build()
{
	Vector3F vp[4];
	vp[0].set(1.1f, .91f, 1.3f);
	vp[1].set(5.27f, 3.1f, 6.1f);
	vp[2].set(7.49f, 6.721f, 4.9f);
	vp[3].set(4.9f, .2f, .2f);
	
	Vector3F vc[4];
	vc[0].set(1.f, 0.f, 0.f);
	vc[1].set(0.5f, 0.1f, 0.f);
	vc[2].set(0.1f, 0.f, .5f);
	vc[3].set(0.1f, .5f, 0.f);
	
	m_tris.insert(createTriangle(vp[0], vp[1], vp[2],
								vc[0], vc[1], vc[2]) );
								
	m_tris.insert(createTriangle(vp[0], vp[2], vp[3],
								vc[0], vc[2], vc[3]) );
	
/// big cell
	BoundingBox b(0.f, 0.f, 0.f,
					8.f, 8.f, 8.f);
					
	TreeProperty::BuildProfile bf;
	bf._maxLeafPrims = 64;
	KdEngine eng;
	KdNTree<cvx::Triangle, KdNode4 > gtr;
	eng.buildTree<cvx::Triangle, KdNode4, 4>(&gtr, &m_tris, b, &bf);
	
	VoxelEngine<cvx::Triangle, KdNode4 >::Profile vf;
	vf._tree = &gtr;
	
	m_engine[0].setBounding(b);
	m_engine[0].build(&vf);
	std::cout<<"\n grid n cell "<<m_engine[0].numCells();
	
/// 512^3 grid
	int level = 6;
	unsigned code = encodeMorton3D(4, 4, 4);
	m_voxels[0].setPos(code, level);
	m_engine[0].extractContours(m_voxels[0]);
	m_engine[0].printContours(m_voxels[0]);
	
	b.setMax(4.f, 4.f, 4.f);
	m_engine[1].setBounding(b);
	m_engine[1].build(&vf);
	
	level = 7;
	code = encodeMorton3D(2, 2, 2);
	m_voxels[1].setPos(code, level);
	m_engine[1].extractContours(m_voxels[1]);
	m_engine[1].printContours(m_voxels[1]);
	
	b.setMin(4.f, 4.f, 4.f);
	b.setMax(8.f, 8.f, 8.f);
	m_engine[2].setBounding(b);
	m_engine[2].build(&vf);
	
	code = encodeMorton3D(6, 6, 6);
	m_voxels[2].setPos(code, level);
	m_engine[2].extractContours(m_voxels[2]);
	m_engine[2].printContours(m_voxels[2]);
	
	b.setMin(4.f, 4.f, 0.f);
	b.setMax(8.f, 8.f, 4.f);
	m_engine[3].setBounding(b);
	m_engine[3].build(&vf);
	
	code = encodeMorton3D(6, 6, 2);
	m_voxels[3].setPos(code, level);
	m_engine[3].extractContours(m_voxels[3]);
	m_engine[3].printContours(m_voxels[3]);
	
	b.setMin(4.f, 0.f, 0.f);
	b.setMax(8.f, 4.f, 4.f);
	m_engine[4].setBounding(b);
	m_engine[4].build(&vf);
	
	code = encodeMorton3D(6, 2, 2);
	m_voxels[4].setPos(code, level);
	m_engine[4].extractContours(m_voxels[4]);
	m_engine[4].printContours(m_voxels[4]);
	
	b.setMin(4.f, 0.f, 4.f);
	b.setMax(8.f, 4.f, 8.f);
	m_engine[5].setBounding(b);
	m_engine[5].build(&vf);
	
	code = encodeMorton3D(6, 2, 6);
	m_voxels[5].setPos(code, level);
	m_engine[5].extractContours(m_voxels[5]);
	m_engine[5].printContours(m_voxels[5]);
	
	buildPyramid();
}

void VoxTest::buildPyramid()
{
	Vector3F nor(0.f, -1.f, -0.32f);
	//nor.normalize();
	Vector3F pnt(-1.55f, -.49f, 0.f);
	float w = -pnt.dot(nor);
/// bottom (0, -.5, 0)
	m_pyramid.plane(0)->set(nor.x, nor.y, nor.z, w);
	
	nor.set(-.8f, 1.f, .56f);
	nor.normalize();
	pnt.set(-2.21f, -.5f, 0.f);
	w = -pnt.dot(nor);
/// -x (-2, -.5, 0)
	m_pyramid.plane(1)->set(nor.x, nor.y, nor.z, w);
	
	nor.set(1.f, 1.f, -0.33f);
	nor.normalize();
	pnt.set(-1.f, -.5f, -.3f);
	w = -pnt.dot(nor);
/// +x (-1, -.5, 0)
	m_pyramid.plane(2)->set(nor.x, nor.y, nor.z, w);
	
	nor.set(0.f, .695f, -.73f);
	nor.normalize();
	pnt.set(-1.5, -.15, -.5);
	w = -pnt.dot(nor);
/// -z (-1.5, -.5, -.5)
	m_pyramid.plane(3)->set(nor.x, nor.y, nor.z, w);
/// +z (-1.5, -.5, .5)
	m_pyramid.plane(4)->set(0.f, .7071f, .7071f, 0.f);
	
	m_pyramid.bbox()->setMin(-2.f, -.5f, -.5f);
	m_pyramid.bbox()->setMax(-1.f, 0.f, .5f);
}

const float VoxTest::TestColor[6][3] = {
{.1f, .1f, .1f},
{0.f, .1f, .5f},
{.9f, .1f, .1f},
{.9f, .9f, .1f},
{0.f, .3f, .5f},
{0.1f, .3f, 0.f}
};

}