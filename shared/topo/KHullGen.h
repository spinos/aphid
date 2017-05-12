/*
 *  KHullGen.h
 *  kmean group convex hull each
 *
 *  Created by jian zhang on 5/13/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_K_HULL_GEN_H
#define APH_K_HULL_GEN_H

#include <sdb/ValGrid.h>
#include <math/kmean.h>
#include <topo/ConvexHullGen.h>
#include <geom/ATriangleMesh.h>

namespace aphid {

template<typename T>
class KHullGen : public sdb::ValGrid<T > {

typedef sdb::ValGrid<T > ParentTyp;

public:
	KHullGen();
	virtual ~KHullGen();
	
	void build(ATriangleMesh * msh, int level, int k);
	
protected:

private:
};

template<typename T>
KHullGen<T>::KHullGen()
{}

template<typename T>
KHullGen<T>::~KHullGen()
{}

template<typename T>
void KHullGen<T>::build(ATriangleMesh * msh, int level, int k)
{
	int n = ParentTyp::numCellsAtLevel(level);
	const int d = 3;
	DenseMatrix<float> kmnd(n, d);
	
	int nv = 0;
	T vcel;
	ParentTyp::begin();
    while(!ParentTyp::end() ) {
        sdb::Coord4 c = ParentTyp::key();
        if(c.w == 3) {
			ParentTyp::getFirstValue(vcel);
			
			kmnd.column(0)[nv] = vcel._pos.x;
			kmnd.column(1)[nv] = vcel._pos.y;
			kmnd.column(2)[nv] = vcel._pos.z;
		
			nv++;
        }
        
        if(c.w > level) {
			break;
		}
			
		ParentTyp::next();
	}
	
	KMeansClustering2<float> cluster;
	cluster.setKND(k, n, d);
	if(!cluster.compute(kmnd) ) {
		std::cout<<"\n kmean failed ";
	}
	
	ConvexHullGen * hlg = new ConvexHullGen[k];
	
	for(int i=0;i<n;++i) {
		Vector3F pe(kmnd.column(0)[i], kmnd.column(1)[i], kmnd.column(2)[i]);
			
		int g = cluster.groupIndices()[i];
		hlg[g].addSample(pe);
		
	}
		
	int nt = 0;
	for(int j = 0;j<k;++j) {
	
		hlg[j].processHull();
		
		nt+= hlg[j].getNumFace();
		
	}
	
	msh->create(nt * 3, nt);
	unsigned * indDst = msh->indices();
    Vector3F * pntDst = msh->points();
    Vector3F * nmlDst = msh->vertexNormals();
    
	nt = 0;
	for(int j = 0;j<k;++j) {
		hlg[j].extractMesh(&pntDst[nt*3], &nmlDst[nt*3], 
				&indDst[nt*3], nt*3);
		nt += hlg[j].getNumFace();
	}
	
	delete[] hlg;
	
}

}

#endif
