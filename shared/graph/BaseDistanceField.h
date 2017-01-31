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
    
/// find edge cross front, assign distance to nodes
    template<typename Tf>
    void findEdgeCross(Tf * intersectF) 
    {
        IDistanceEdge * egs = edges();
        DistanceNode * nds = nodes();
        
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
                const float d1 = crossp.distanceTo(node1.pos);
                if(node1.val > d1) {
                    node1.val = d1;
                }
                const float d2 = crossp.distanceTo(node2.pos);
                if(node2.val > d2) {
                    node2.val = d2;
                }
                e.cx = d1 / (d1 + d2);
            }
            
        }
    }
    
protected:

private:

};

}

#endif
