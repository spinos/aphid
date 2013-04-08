/*
 *  ControlGraph.cpp
 *  hc
 *
 *  Created by jian zhang on 4/8/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "ControlGraph.h"

ControlGraph::ControlGraph()
{
	m_intersectCtx = new RayIntersectionContext;
	m_intersectCtx->setComponentFilterType(PrimitiveFilter::TFace);
	
	TargetGraph * g0 = new TargetGraph;
	simpleGraph(g0, -15, 10, 10);
	g0->setControlId(0);
	g0->reset();
	m_graphList.push_back(g0);
	
	TargetGraph * g1 = new TargetGraph;
	simpleGraph(g1, -5, 20, 10);
	g1->setControlId(1);
	g1->reset();
	m_graphList.push_back(g1);
	
	TargetGraph * g2 = new TargetGraph;
	simpleGraph(g2, 5, 20, 10);
	g2->setControlId(2);
	g2->reset();
	m_graphList.push_back(g2);
	
	TargetGraph * g3 = new TargetGraph;
	simpleGraph(g3, 15, 10, 10);
	g3->setControlId(3);
	g3->reset();
	m_graphList.push_back(g3);
	
	TargetGraph * g4 = new TargetGraph;
	simpleGraph(g4, 5, 0, 10);
	g4->setControlId(4);
	g4->reset();
	m_graphList.push_back(g4);
	
	TargetGraph * g5 = new TargetGraph;
	simpleGraph(g5, -5, 0, 10);
	g5->setControlId(5);
	g5->reset();
	m_graphList.push_back(g5);
	
	TargetGraph * g6 = new TargetGraph;
	simpleGraph(g6, -22, 32, 10);
	g6->initCoords();
	g6->setControlId(6);
	g6->reset();
	m_graphList.push_back(g6);
	
	TargetGraph * g7 = new TargetGraph;
	simpleGraph(g7, 22, 32, 10);
	g7->setControlId(7);
	g7->reset();
	m_graphList.push_back(g7);
}

TargetGraph * ControlGraph::firstGraph()
{
	m_graphIt = m_graphList.begin();
	return *m_graphIt;
}

TargetGraph * ControlGraph::nextGraph()
{
	m_graphIt++;
	if(!hasGraph()) return 0;
	return *m_graphIt;
}

bool ControlGraph::hasGraph()
{
	return m_graphIt != m_graphList.end();
}

bool ControlGraph::pickupControl(const Ray & ray, Vector3F & hit)
{
	m_intersectCtx->reset();
	m_currentGraph = 0;
	for(m_graphIt = m_graphList.begin(); m_graphIt != m_graphList.end(); ++m_graphIt) {
		if((*m_graphIt)->intersect(ray, m_intersectCtx)) {
			hit = m_intersectCtx->m_hitP;
			m_currentGraph = *m_graphIt;
			return true;
		}
	}
	return false;
}

void ControlGraph::updateControl()
{
	if(!m_currentGraph) return;
	m_currentGraph->computeWeight(m_intersectCtx->m_componentIdx, m_intersectCtx->m_hitP);
}

void ControlGraph::simpleGraph(TargetGraph * g, float tx, float ty, float tz)
{
	g->createVertices(3);
	g->createIndices(3);
	g->createTargetIndices(3);
	g->createVertexWeights(3);
	g->setVertex(0, 0, 0, 0);
	g->setVertex(1, 7, 0, 0);
	g->setVertex(2, 7, 7, 0); 
	g->setTriangle(0, 0, 1, 2);
	g->setTarget(0, 0);
	g->setTarget(1, 1);
	g->setTarget(2, 2);
	g->move(tx, ty, tz);
	g->initCoords();
	
}

unsigned ControlGraph::handleId() const
{
	return m_currentGraph->getControlId();
}

TargetGraph * ControlGraph::activeGraph() const
{
	return m_currentGraph;
}

TargetGraph * ControlGraph::getGraph(unsigned idx) const
{
	if(idx >= m_graphList.size()) return 0;
	return m_graphList[idx];
}
