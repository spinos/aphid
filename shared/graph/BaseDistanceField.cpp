/*
 *  BaseDistanceField.cpp
 *  
 *
 *  Created by zhang on 17-1-31.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "BaseDistanceField.h"

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
		if(vj == i) {
			vj = eg.vi.y;
        }
            
		DistanceNode & B = nodes()[vj];
		if(B.stat == sdf::StUnknown) {
		
/// min distance to B via A
/// need eikonal approximation here
			if(B.val > A.val + eg.len) {
				B.val = A.val + eg.len;
            }
				
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

void BaseDistanceField::marchOutside(const int & originNodeInd)
{
    unvisitAllNodes();
        
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

 float BaseDistanceField::getShortestCutEdgeLength(const int & idx) const
 {
    float r = 1e8f;
    const DistanceNode & A = nodes()[idx];
    const int endj = edgeBegins()[idx+1];
	int vj, j = edgeBegins()[idx];
	for(;j<endj;++j) {
		
		int k = edgeIndices()[j];
		const IDistanceEdge & eg = edges()[k];
		
		if(eg.cx > -1.f) {
			if(r > eg.len) {
				r = eg.len;
            }
		}
	}
    return r;
 }

void BaseDistanceField::expandFrontEdge()
{
    DistanceNode * nds = nodes();
    const int & nv = numNodes();
    for(int i = 0;i<nv;++i) {
        float l = getShortestCutEdgeLength(i);
        if(l < 1e7f) {
            nds[i].val -= l;
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

float BaseDistanceField::distanceToFront2(int & closestEdgeIdx,
                                const int & idx) const
{
    const DistanceNode & d = nodes()[idx];
    
    float closestCut = 2.f;
    float currentCut = 3.f;
    
/// for each neighbor of A find closest cut
	const int endj = edgeBegins()[idx+1];
	int vj, j = edgeBegins()[idx];
	for(;j<endj;++j) {
		
		int k = edgeIndices()[j];

		const IDistanceEdge & eg = edges()[k];
        
        vj = eg.vi.x;
		if(vj == idx) {
			vj = eg.vi.y;
        }
        
/// sign changes
		if(nodes()[vj].val * d.val < 0.f) {
            currentCut = Absolute<float>(d.val) / (Absolute<float>(d.val) +
                                                    Absolute<float>(nodes()[vj].val ) );
            
            if(closestCut > currentCut) {
                closestEdgeIdx = k;
                closestCut = currentCut;
            }
        }
	}
    
    return closestCut;
}

void BaseDistanceField::moveToFront2(const int & idx,
                            const int & edgeIdx)
{    
    DistanceNode & d = nodes()[idx];
    
    IDistanceEdge & ce = edges()[edgeIdx];
    const DistanceNode & va = nodes()[ce.vi.x];
    const DistanceNode & vb = nodes()[ce.vi.y];
    
    Vector3F dv = vb.pos - va.pos;
    if(idx == ce.vi.y) {
        dv.reverse();
    }
    
    const float l = Absolute<float>(va.val) + Absolute<float>(vb.val);
    
    d.pos += dv * Absolute<float>(d.val) / l;
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
        float d = distanceToFront2(iedge, i);
        if(d < threshold) {
            moveToFront2(i, iedge);
        }
		
	}
}

void BaseDistanceField::setNodeDistance(const int & idx,
                        const float & v) 
{
    DistanceNode & d = nodes()[idx];
    if(d.stat != sdf::StKnown) {
        d.val = v;
        d.stat = sdf::StKnown;
    } else {
        if(Absolute<float>(d.val) > Absolute<float>(v) ) {
            d.val = v;
        }
    }
    
}

void BaseDistanceField::uncutEdges()
{
    IDistanceEdge * egs = edges();
    const int & ne = numEdges();
    for(int i=0;i<ne;++i) {
        IDistanceEdge & e = egs[i];
        e.cx = -1.f;
    }
}

void BaseDistanceField::cutEdge(const int & v1, const int & v2,
                const float & d1, const float & d2)
{
    if(d1 * d2 >= 0.f) {
        return;
    }
    
    IDistanceEdge * e = edge(v1, v2);
    if(!e) {
        return;
    }
    
    const sdb::Coord2 & k = e->vi;

    if(k.x == v1) {
        e->cx = Absolute<float>(d1) / 
                    (Absolute<float>(d1) + Absolute<float>(d2) );
    } else {
        e->cx = Absolute<float>(d2) / 
                    (Absolute<float>(d1) + Absolute<float>(d2) );
    }
}

void BaseDistanceField::cutEdges()
{
    DistanceNode * nds = nodes();
    IDistanceEdge * egs = edges();
    const int & ne = numEdges();
    for(int i=0;i<ne;++i) {
        IDistanceEdge & e = egs[i];
        e.cx = -1.f;
        const sdb::Coord2 & k = e.vi;
        DistanceNode & node1 = nds[k.x];
        DistanceNode & node2 = nds[k.y];
        if(node1.stat == sdf::StKnown && 
            node2.stat == sdf::StKnown) {
            if(node1.val * node2.val < 0.f) {
                e.cx = Absolute<float>(node1.val) / 
                    (Absolute<float>(node1.val) + Absolute<float>(node2.val) );
            } 
        }
    }
}

}
