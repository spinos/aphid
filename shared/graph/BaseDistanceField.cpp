/*
 *  BaseDistanceField.cpp
 *  
 *
 *  Created by zhang on 17-1-31.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "BaseDistanceField.h"
#include <math/Ray.h>

namespace aphid {

BaseDistanceField::BaseDistanceField()
{}

BaseDistanceField::~BaseDistanceField()
{}

void BaseDistanceField::resetNodes(float val, sdf::NodeState lab, sdf::NodeState stat)
{
    const int & n = numNodes();
	for(int i = 0;i<n;++i) {
		DistanceNode & d = nodes()[i];
        d.val = val;
		d.label = lab;
		d.stat = stat;
	}
}

void BaseDistanceField::unvisitAllNodes()
{
    const int & n = numNodes();
	for(int i = 0;i<n;++i) {
		DistanceNode & d = nodes()[i];
        d.stat = sdf::StFar;
	}
}

void BaseDistanceField::fastMarchingMethod()
{
/// heap of trial
	std::map<int, int> trials;
	const int n = numNodes();
	int i = 0;
	for(;i<n;++i) {
		DistanceNode & d = nodes()[i];
		if(d.stat == sdf::StKnown) {
			propagate(trials, i);
		}
	}
	
/// for each trial
	while (trials.size() > 0) {

/// A is first in trial		
		i = trials.begin()->first;
/// distance is known after propagation
		nodes()[i].stat = sdf::StKnown;
/// remove A from trial
		trials.erase(trials.begin() );
		
/// from A
		propagate(trials, i);
		
		//std::cout<<"\n trial n "<<trials.size();
		//std::cout.flush();
	}
}

/// A to B
void BaseDistanceField::propagate(std::map<int, int > & heap, 
												const int & i)
{
	const DistanceNode & A = nodes()[i];
	
/// for each neighbor of A
	const int endj = edgeBegins()[i+1];
	int vj, j = edgeBegins()[i];
	for(;j<endj;++j) {
		
		int k = edgeIndices()[j];

		const IDistanceEdge & eg = edges()[k];
		
		vj = eg.vi.x;
		if(vj == i)
			vj = eg.vi.y;
			
		DistanceNode & B = nodes()[vj];
		if(B.stat == sdf::StUnknown) {
		
/// min distance to B via A
/// need eikonal approximation here
			if(A.val + eg.len < B.val)
				B.val = A.val + eg.len;
				
/// add to trial
			heap[vj] = 0;
		}
	}
}

int BaseDistanceField::nodeFarthestFrom(const Vector3F & origin,
                        const Vector3F & dir) const
{
    float maxD = 0.f;
    float dist;
    int ri = 0;
    const int & n = numNodes();
	for(int i = 0;i<n;++i) {
		const DistanceNode & d = nodes()[i];
        dist = (d.pos - origin).dot(dir);
        if(maxD < dist) {
            maxD = dist;
            ri = i;
        }
	}
    return ri;
}

void BaseDistanceField::setFarNodeInside()
{
	const int n = numNodes();
	for(int i = 0;i<n;++i) {
		DistanceNode & d = nodes()[i];
		if(d.stat == sdf::StFar) {
/// inside distance is negative
			if(d.val > 0.f)
				d.val = -d.val;
		}
	}
}

void BaseDistanceField::expandFront(const float & x)
{
    const int n = numNodes();
	for(int i = 0;i<n;++i) {
		DistanceNode & d = nodes()[i];
		d.val -= x;
	}
}

void BaseDistanceField::marchOutside(const int & originNodeInd)
{
	int i = originNodeInd;
	
/// heap of trial
	std::map<int, int> trials;
	trials[i] = 0;
	
/// for each trial
	while (trials.size() > 0) {

/// A is first in trial		
		i = trials.begin()->first;

		nodes()[i].stat = sdf::StVisited;
/// remove A from trial
		trials.erase(trials.begin() );
		
/// from A
		propagateVisit(trials, i);
		
		//std::cout<<"\n trial n "<<trials.size();
		//std::cout.flush();
	}
	
}

void BaseDistanceField::propagateVisit(std::map<int, int > & heap, const int & i)
{
	const DistanceNode & A = nodes()[i];
	
/// for each neighbor of A
	const int endj = edgeBegins()[i+1];
	int vj, j = edgeBegins()[i];
	for(;j<endj;++j) {
		
		int k = edgeIndices()[j];

		const IDistanceEdge & eg = edges()[k];
		
		vj = eg.vi.x;
		if(vj == i)
			vj = eg.vi.y;
			
		DistanceNode & B = nodes()[vj];
/// do not cross front
		if(eg.cx < 0.f) {
/// do not visit inside
			if( B.val > 0.f && B.stat == sdf::StFar) {
				heap[vj] = 0;
            }
		}
	}
}

float BaseDistanceField::distanceToFront(int & closestEdgeIdx,
                                const int & idx) const
{
    const DistanceNode & d = nodes()[idx];
    
    float closestCut = 2.f;
    float currentCut = 3.f;
    
/// for each neighbor of A find closest cut
	const int endj = edgeBegins()[idx+1];
	int j = edgeBegins()[idx];
	for(;j<endj;++j) {
		
		int k = edgeIndices()[j];

		const IDistanceEdge & eg = edges()[k];
		if(eg.cx > 0.f) {
/// from A
            if(eg.vi.x == idx) {
                currentCut = eg.cx;
            } else {
                currentCut = 1.f - eg.cx;
                if(currentCut < 1e-3f) {
                    currentCut = 0.f;
                }
            }
            
            if(closestCut > currentCut) {
                closestEdgeIdx = k;
                closestCut = currentCut;
            }
        }
	}
    
    return closestCut;
}


void BaseDistanceField::moveToFront(const int & idx,
                            const int & edgeIdx)
{    
    DistanceNode & d = nodes()[idx];
    
    const IDistanceEdge & ce = edges()[edgeIdx];
    const Vector3F dv = nodes()[ce.vi.y].pos - nodes()[ce.vi.x].pos;
    if(ce.vi.x == idx) {        
       d.pos += dv * ce.cx;
    } else {
       d.pos += dv * (ce.cx - 1.f);
    }
    
    d.val = 0.f;
    
    const int endj = edgeBegins()[idx+1];
	int j = edgeBegins()[idx];
	for(;j<endj;++j) {
		
		int k = edgeIndices()[j];

        IDistanceEdge & eg = edges()[k];
		eg.cx = -1.f;
	}
    
}

void BaseDistanceField::snapToFront(const float & threshold)
{
    int iedge;
    const int n = numNodes();
	for(int i = 0;i<n;++i) {
        float d = distanceToFront(iedge, i);
        if(d < threshold) {
            moveToFront(i, iedge);
        }
		
	}
}

}
