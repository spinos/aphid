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
	
	TargetGraph * graph = new TargetGraph;
	graph->createVertices(3);
	graph->createIndices(3);
	graph->createTargetIndices(3);
	graph->createVertexWeights(3);
	graph->setVertex(0, 0, 0, 0);
	graph->setVertex(1, 4, 0, 0);
	graph->setVertex(2, 4, 4, 0); 
	graph->move(20, 20, 10);
	graph->setTriangle(0, 0, 1, 2);
	graph->setTargetTriangle(0, 0, 1, 1);
	graph->initCoords();
	graph->reset();
	
	m_graphList.push_back(graph);
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
