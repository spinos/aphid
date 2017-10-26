/*
 *  GeodesicRegion.cpp
 *  
 *
 *  Created by jian zhang on 10/28/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "GeodesicRegion.h"

namespace aphid {

namespace topo {

GeodesicRegion::GeodesicRegion()
{}

GeodesicRegion::~GeodesicRegion()
{}

void GeodesicRegion::createFromTriangles(const int& vertexCount,
				const float* vertexPos,
				const float* vertexNml,
				const int& triangleCount,
				const int* triangleIndices)
{
	GeodesicDistance::buildTriangleGraph(vertexCount,
				vertexPos, vertexNml,
				triangleCount, triangleIndices);
	DistancePath::create(vertexCount);
	
}

bool GeodesicRegion::findRootNode()
{
	const int ns = numSeeds();
	if(ns<1 )
		return false;
	
	const int& nv = numVertices();
	float* sumd = new float[nv];
	memset(sumd, 0, nv << 2);
	
	for(int i=0;i<ns;++i) {
	
		float* d2si = distanceToSeed(i);
		calaculateDistanceTo(d2si, seedNodeIndex(i) );
				
		for(int j=0;j<nv;++j) {
/// could be unknown
			if(d2si[j] < 1e19f)
				sumd[j] += d2si[j];
		}
		
	}
	
	float maxd = -1e9f;
	int ri = 0;
	for(int i=0;i<nv;++i) {
		if(maxd < sumd[i]) {
			maxd = sumd[i];
			ri = i;
		}
	}
	
	delete[] sumd;
	
	setRootNodeIndex(ri);
	calaculateDistanceTo(distanceToRoot(), ri );
	
	std::cout<<"\n root node "<<ri;
	std::cout.flush();

	return true;
}

void GeodesicRegion::growRegions()
{
	labelRootAndSeedPoints();
	propagateLabels();
	colorByRegionLabels();
}

void GeodesicRegion::propagateLabels()
{
	resetNodes(1e20f, sdf::StBackGround, sdf::StUnknown);
	uncutEdges();
	
	const int ns = numRegions();
	for(int i=0;i<ns;++i) {
		DistanceNode & d = nodes()[siteNodeIndex(i)];
		d.val = 0.f;
		d.stat = sdf::StKnown;
	}
	
/// heap of trial
	std::deque<int> trials;
	const int n = numNodes();
	for(int i=0;i<n;++i) {
		DistanceNode & d = nodes()[i];
		if(d.stat == sdf::StKnown) {
			propagateLabels(trials, i);
		}
	}
	
/// for each trial
	while (trials.size() > 0) {

/// A is first in trial		
		int i = trials[0];
/// from A
		propagateLabels(trials, i);
		
/// distance is known after propagation
		nodes()[i].stat = sdf::StKnown;

/// remove A from trial
		trials.erase(trials.begin() );	

	}
		
}

void GeodesicRegion::propagateLabels(std::deque<int > & heap, 
												const int & i)
{
	int* labs = vertexLabels();
	const int& la = labs[i];
	const float& da = distanceToSite(la)[i];
	const DistanceNode & A = nodes()[i];
	
/// for each neighbor of A
	const int& endj = edgeBegins()[i+1];
	int vj, j = edgeBegins()[i];
	for(;j<endj;++j) {
		
		int k = edgeIndices()[j];

		const IDistanceEdge & eg = edges()[k];
		
		vj = eg.vi.x;
		if(vj == i) {
			vj = eg.vi.y;
        }
            
		DistanceNode& B = nodes()[vj];
		
		bool stat = (B.stat == sdf::StUnknown);
		if(!stat) {
			const int lb = labs[vj];
			if(la != lb)
				stat = distanceToSite(lb)[vj] > (da + eg.len);
		}
		
		if(stat) {
/// assign label A to B
			labs[vj] = la;
/// add to trial
			addNodeToHeap(heap, vj);
			
		}
	}
}

}

}