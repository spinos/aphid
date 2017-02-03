/*
 *  BaseDistanceField.h
 *  
 *
 *  Created by zhang on 17-1-31.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_GRAPH_BASE_DISTANCE_FIELD_H
#define APH_GRAPH_BASE_DISTANCE_FIELD_H

#include "AGraph.h"
#include <map>
#include <math/miscfuncs.h>
#include <math/Ray.h>

namespace aphid {

namespace sdf {
enum NodeState {
	StBackGround = 0,
	StFront = 1,
	StUnknown = 2,
	StKnown = 3,
	StFar = 4,
	StVisited = 5
};

}

struct DistanceNode {
	
	Vector3F pos;
	float val;
	short label;
	short stat;
};

struct IDistanceEdge {

	sdb::Coord2 vi; /// ind to node
	float len; /// length of edge
	float err; /// max error of distance always positive
	float cx; /// x [0,1] where front cross
};

class BaseDistanceField : public AGraph<DistanceNode, IDistanceEdge > {

public:
    BaseDistanceField();
    virtual ~BaseDistanceField();
    
    template<typename Tf>
    void calculateDistance(Tf * intersectF) 
    {
        resetNodes(1e20f, sdf::StBackGround, sdf::StUnknown);
        
        IDistanceEdge * egs = edges();
        DistanceNode * nds = nodes();
        
        Vector3F agp, agn;
        intersectF->getAggregatedPositionNormal(agp, agn);
        
/// find edge cross front, assign distance to connected nodes        
        Vector3F por;
        const int & ne = numEdges();
        for(int i=0;i<ne;++i) {
            IDistanceEdge & e = egs[i];
            const sdb::Coord2 & k = e.vi;
            DistanceNode & node1 = nds[k.x];
            DistanceNode & node2 = nds[k.y];
            
            bool rpos = true;
            Ray r(node1.pos, node2.pos);
/// from outside to inside
            if(r.m_dir.dot(agn) > 0.f ) {
                r.m_origin = node2.pos;
                r.m_dir.reverse();
                rpos = false;
            }
           
            if(intersectF->rayIntersect(r) ) {
                const Vector3F & crossp = intersectF->rayIntersectPoint();
                float d1, d2;
                if(rpos) {
                    d1 = crossp.distanceTo(node1.pos);
                    d2 = e.len - d1;
                
                } else {
                    d2 = crossp.distanceTo(node2.pos);
                    d1 = e.len - d2;
                }
                
                e.cx = d1 / e.len;
                
                if(Absolute<float>(node1.val) > d1 ) {
                    node1.val = d1;
                    node1.stat = sdf::StKnown;
                }
                
                if(Absolute<float>(node2.val) > d2 ) {
                    node2.val = d2;
                    node2.stat = sdf::StKnown;
                }
                
                if(r.m_dir.dot(agn ) < -.5f) {
                if(rpos) {
                    if(node2.val > 0.f) {
                        node2.val *= -1.f;
                    }
                    
                    
                } else {
                    if(node1.val > 0.f) {
                        node1.val *= -1.f;
                    }
                    
                }
                }
                
            } else {
                e.cx = -1.f;
            }
            
        }

/// propagate distance to all nodes        
        fastMarchingMethod();
        
        int iFar = nodeFarthestFrom(agp, agn);
/// visit out nodes
        marchOutside(iFar);
/// unvisited nodes are inside
        setFarNodeInside();
/// merge short edges
        snapToFront();
        
    }
    
protected:	
	void resetNodes(float val, sdf::NodeState lab, sdf::NodeState stat);
    void unvisitAllNodes();
    void fastMarchingMethod();
	void marchOutside(const int & originNodeInd);
    void setFarNodeInside();
    void propagateVisit(std::map<int, int > & heap, const int & i);
/// for each node connected to cut edge
/// val minus shortest cut edge length
	void expandFrontEdge();
/// if node connected to edge cut close to 0 or 1
/// un-cut all edges connected and set node val zero
    void snapToFront(const float & threshold = .19f);
    
private:
/// propagate distance value
    void propagate(std::map<int, int > & heap, const int & i);
	int nodeFarthestFrom(const Vector3F & origin,
                        const Vector3F & dir) const;
/// lowest edge cut connected to node
    float distanceToFront(int & closestEdgeIdx,
                const int & idx) const;
/// move node to front and un-cut all connected edges
    void moveToFront(const int & idx,
                const int & edgeIdx);
    float distanceToFront2(int & closestEdgeIdx,
                const int & idx) const;
    void moveToFront2(const int & idx,
                const int & edgeIdx);
    float getShortestCutEdgeLength(const int & idx) const;
        
};

}

#endif
