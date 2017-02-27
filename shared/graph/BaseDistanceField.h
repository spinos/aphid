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

struct CalcDistanceProfile {
	Vector3F referencePoint;
	float offset;
	Vector3F direction;
	float snapDistance;
};

class BaseDistanceField : public AGraph<DistanceNode, IDistanceEdge > {

public:
    BaseDistanceField();
    virtual ~BaseDistanceField();
    
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
    void snapToFront(const float & threshold = .2f);
    void snapToFrontByDistance(const float & threshold = .1f);
    int nodeFarthestFrom(const Vector3F & origin,
                        const Vector3F & dir) const;
    void setNodeDistance(const int & idx,
                        const float & v);
                        
    void uncutEdges();
/// when sign changes
    void cutEdge(const int & v1, const int & v2,
                const float & d1, const float & d2);
    void cutEdges();
/// move idx-th node to pos, uncut connected edges
	void moveNodeToFront(const Vector3F & pos,
						const int & idx);
/// longest edge connected to i-th node
	float longestEdgeLength(const int & idx) const;
	
/// set node distance known
/// cut edges move to front if necessary
	void setNodePosDistance(const Vector3F & pos,
						const float & v, 
						const int & idx);
						
private:
/// propagate distance value
    void propagate(std::map<int, int > & heap, const int & i);
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
	float closestDistanceToFront(int & closestEdgeIdx,
                const int & idx) const;
	void moveToFront3(const int & idx);
	void cutEdgesConnectedToNode(const int & idx);
						
};

}

#endif
