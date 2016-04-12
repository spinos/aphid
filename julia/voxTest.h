/*
 *  voxTest.h
 *  
 *
 *  Created by jian zhang on 4/13/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef VOX_TEST_H
#define VOX_TEST_H
#include <KdEngine.h>
#include <VoxelEngine.h>
#include <ConvexShape.h>

namespace aphid {

class VoxTest {

public:
	void build();
	
	cvx::Triangle createTriangle(const Vector3F & p0,
								const Vector3F & p1,
								const Vector3F & p2,
								const Vector3F & c0,
								const Vector3F & c1,
								const Vector3F & c2);
	sdb::VectorArray<cvx::Triangle> m_tris;
	
	VoxelEngine<cvx::Triangle, KdNode4 > m_engine[6];	
	Voxel m_voxels[6];
	
	static const float TestColor[6][3];
};

}

#endif