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

namespace aphid {

class Ray;

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
        
/// find edge cross front, assign distance to connected nodes        
        Vector3F por;
        const int & ne = numEdges();
        for(int i=0;i<ne;++i) {
            IDistanceEdge & e = egs[i];
            const sdb::Coord2 & k = e.vi;
            DistanceNode & node1 = nds[k.x];
            DistanceNode & node2 = nds[k.y];
            
            Ray r(node1.pos, node2.pos);
           
            if(intersectF->rayIntersect(r) ) {
                const Vector3F & crossp = intersectF->rayIntersectPoint();
                float d1 = crossp.distanceTo(node1.pos);
                float d2 = e.len - d1;
                
                if(node1.val > d1) {
                    node1.val = d1;
                    node1.stat = sdf::StKnown;
                }
                
                if(node2.val > d2) {
                    node2.val = d2;
                    node2.stat = sdf::StKnown;
                }
                e.cx = d1 / e.len;
            } else {
                e.cx = -1.f;
            }
            
        }
        
/// propagate distance to all nodes        
        fastMarchingMethod();
        
        //expandFront(1.f);
        
        unvisitAllNodes();
        
        Vector3F agp, agn;
        intersectF->getAggregatedPositionNormal(agp, agn);
        
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
	void expandFront(const float & x);
/// if node connected to edge cut close to 0 or 1
/// un-cut all edges connected and set node val zero
    void snapToFront(const float & threshold = .17f);
    
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
        
};

}

#endif
