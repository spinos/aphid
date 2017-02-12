/*
 *  LodGridMesher.h
 *  
 *  T as grid type, Tn as node type
 *  one piece per node
 *
 *  Created by jian zhang on 2/12/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_SDB_LOD_GRID_MESHER_H
#define APH_SDB_LOD_GRID_MESHER_H

#include <geom/ATriangleMesh.h>
#include <math/miscfuncs.h>
#include <math/Matrix33F.h>

namespace aphid {

namespace sdb {

template<typename T, typename Tn>
class LodGridMesher {

	T * m_grid;
	
public:
	LodGridMesher(T * grid);
	
	void buildMesh(ATriangleMesh * mesh,
					int level);
	
protected:

private:
	void randomTriangleAt(Vector3F & p1, Vector3F & p2, Vector3F & p3,
			Vector3F & n1, Vector3F & n2, Vector3F & n3,
			const Tn & samp,
			const float & radius);
	
};

template<typename T, typename Tn>
LodGridMesher<T, Tn>::LodGridMesher(T * grid)
{ m_grid = grid; }

template<typename T, typename Tn>
void LodGridMesher<T, Tn>::buildMesh(ATriangleMesh * mesh,
					int level)
{
	const int nt = m_grid->countLevelNodes(level);
	const int nv = nt * 3;
	mesh->create(nv, nt);
	
	const float radius = m_grid->levelCellSize(level + 1) * .4f;
	
	Tn * samps = new Tn[nv];
	m_grid->dumpLevelNodes(samps, level);
	
	Vector3F * pos = mesh->points();
	Vector3F * nml = mesh->vertexNormals();
	unsigned * facevs = mesh->indices();
	
	for(int i=0;i<nt;++i) {
		facevs[i*3] = i*3;
		facevs[i*3 + 1] = i*3 + 1;
		facevs[i*3 + 2] = i*3 + 2;
		
		randomTriangleAt(pos[i*3], pos[i*3 + 1], pos[i*3 + 2], 
			nml[i*3], nml[i*3 + 1], nml[i*3 + 2], 
			samps[i],
			radius);
		
	}
	
	delete[] samps;
	
}

template<typename T, typename Tn>
void LodGridMesher<T, Tn>::randomTriangleAt(Vector3F & p1, Vector3F & p2, Vector3F & p3,
			Vector3F & n1, Vector3F & n2, Vector3F & n3,
			const Tn & samp,
			const float & radius)
{
	const Vector3F & sampp = samp.pos;
	const Vector3F & sampn = samp.nml;
	n1 = sampn;
	n2 = sampn;
	n3 = sampn;
	
	const Vector3F s = sampn.perpendicular();
	const Vector3F f = s.cross(sampn);
	Matrix33F mbase(s, sampn, f);
	
	float ang = RandomF01() * 3.f;
	Quaternion qrot(ang, sampn);
	Matrix33F mrot(qrot);
	
	mbase *= mrot;
	
	Vector3F ofs;
	mbase.getSide(ofs);
	
	Vector3F ofsf;
	mbase.getFront(ofsf);
	
	p1 = sampp + ofs * radius;
	
	p2 = sampp - ofs * .5f * radius - ofsf * .768f * radius;
	
	p3 = sampp - ofs * .5f * radius + ofsf * .768f * radius;
}

}

}

#endif

