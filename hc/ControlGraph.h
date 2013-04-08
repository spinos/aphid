/*
 *  ControlGraph.h
 *  hc
 *
 *  Created by jian zhang on 4/8/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <RayIntersectionContext.h>
#include <TargetGraph.h>
#include <vector>
class ControlGraph {
public:
	ControlGraph();
	
	TargetGraph * firstGraph();
	TargetGraph * nextGraph();
	bool hasGraph();
	
	bool pickupControl(const Ray & ray, Vector3F & hit);
	void updateControl();
	
	unsigned handleId() const;
	TargetGraph * activeGraph() const;
	TargetGraph * getGraph(unsigned idx) const;
private:
	void simpleGraph(TargetGraph * g, float tx, float ty, float tz);
	RayIntersectionContext * m_intersectCtx;
	std::vector<TargetGraph *> m_graphList;
	std::vector<TargetGraph *>::iterator m_graphIt;
	TargetGraph * m_currentGraph;
};