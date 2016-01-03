/*
 *  QuatJulia.cpp
 *  
 *
 *  Created by jian zhang on 1/1/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "QuatJulia.h"

namespace jul {

Float4 quatProd(const Float4 & a, const Float4 & b)
{
	return Float4(
			a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w,
			a.x * b.y + a.y * b.x + a.z * b.w - a.w * b.z,
			a.x * b.z - a.y * b.w + a.z * b.x + a.w * b.y,
			a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x);
}

Float4 quatSq(const Float4 & a)
{
	return Float4(
			a.x * a.x - a.y * a.y - a.z * a.z - a.w * a.w,
			2.0 * a.x * a.y,
			2.0 * a.x * a.z,
			2.0 * a.x * a.w);
}

QuatJulia::QuatJulia() 
{
	sdb::TreeNode::MaxNumKeysPerNode = 128;
	sdb::TreeNode::MinNumKeysPerNode = 16;

	m_c = Float4(0.f, 0.f, 0.f, 0.f);
	m_numIter = 8;
	m_numGrid = 200;
	m_tree = new sdb::WorldGrid<sdb::VectorArray<Vector3F >, Vector3F >;
	m_tree->setGridSize(1.f / 11.f);
	generate();
}

QuatJulia::~QuatJulia()
{ delete m_tree; }

void QuatJulia::generate()
{
/// eval at uniform grid
	int i, j, k;
	const float grid = 1.f / (float)m_numGrid;
	const Vector3F origin(grid * .5f, grid * .5f, grid * .5f);
	int n = 0;
	for(k=0; k<m_numGrid; ++k ) {
		for(j=0; j<m_numGrid; ++j ) {
			for(i=0; i<m_numGrid; ++i ) {
				Vector3F sample = origin + Vector3F(grid * i, grid * j, grid * k);
				if( evalAt( sample ) > 0.f ) {
					n++;
					m_tree->insert((const float *)&sample, sample);
				}
			}
		}
	}
	std::cout<<"\n n pnt "<<n
	<<"n grid "<<m_tree->size();
}

float QuatJulia::evalAt(const Vector3F & at) const
{
/// Quaternion Julia Fractals
	Float4 z(at.x, at.y, at.z, 0.f);
	Float4 z2(1.f, 0.f, 0.f, 0.f);

	float n = 0.0;
	float sqr_abs_z = 0.0;

	while (n < m_numIter)
	{
		z2 = quatProd(z, z2) * 2.f;
		z  = quatSq(z) + m_c;

		sqr_abs_z = z.dot(z);
		if (sqr_abs_z >= 4.f)
			break;

		n++;
	}

	return sqr_abs_z - 4.f;
}

}