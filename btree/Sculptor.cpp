/*
 *  Sculptor.cpp
 *  btree
 *
 *  Created by jian zhang on 5/10/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "Sculptor.h"
namespace sdb {

Sculptor::ActiveGroup::ActiveGroup() 
{ 
	vertices = new Ordered<int, VertexP>; 
	reset(); 
}

void Sculptor::ActiveGroup::reset() 
{
	depthMin = 10e8;
	depthMax = -10e8;
	vertices->clear();
	meanPosition.setZero();
	meanNormal.setZero();
	numActivePoints = 0;
	numActiveBlocks = 0;
}

int Sculptor::ActiveGroup::numSelected() { return vertices->numElements(); }

float Sculptor::ActiveGroup::depthRange() 
{
	return depthMax - depthMin;
}

void Sculptor::ActiveGroup::updateDepthRange(const float & d) 
{
	if(d > depthMax) depthMax = d;
	if(d < depthMin) depthMin = d;
}

void Sculptor::ActiveGroup::finish() 
{
	if(vertices->size() < 1) return;
	const float mxr = threshold * 2.f;

	vertices->begin();
	while(!vertices->end()) {
		const List<VertexP> * vs = vertices->value();
		if((vertices->key() - 1) * gridSize - depthMin > mxr) break;
		
		average(vs);
		
		numActiveBlocks++;
		vertices->next();
	}
	
	if(numActivePoints > 0) {
		meanPosition *= 1.f / (float)numActivePoints;
		meanNormal.normalize();
	}
}

void Sculptor::ActiveGroup::average(const List<VertexP> * d)
{
	const int num = d->size();
	if(num < 1) return;
	for(int i = 0; i < num; i++) {
		Vector3F * p = d->value(i).index->t1;
		Vector3F * n = d->value(i).index->t2;
		meanPosition += *p;
		meanNormal += *n;
		numActivePoints++;
	}
}
		
Sculptor::Sculptor() 
{
	m_tree = new C3Tree;
	m_active = new ActiveGroup;
}

Sculptor::~Sculptor()
{
	delete m_tree;
	delete m_active;
}

void Sculptor::beginAddVertices(const float & gridSize)
{
	m_tree->clear();
	m_tree->setGridSize(gridSize);
}

void Sculptor::addVertex(const VertexP & v)
{
	m_tree->insert(v);
}

void Sculptor::endAddVertices()
{
	std::cout<<"grid count "<<m_tree->size();
	m_tree->calculateBBox();
	m_march.initialize(m_tree->boundingBox(), m_tree->gridSize());
}

void Sculptor::setSelectRadius(const float & x)
{
	m_active->threshold = x;
	m_active->gridSize = x * .25f;
}

void Sculptor::selectPoints(const Ray * incident)
{
	m_active->reset(); 
	
	Sequence<Coord3> added;
	if(!m_march.begin(*incident)) return;
	while(!m_march.end()) {
		const std::deque<Vector3F> coords = m_march.touched(m_active->gridSize);
		std::deque<Vector3F>::const_iterator it = coords.begin();
		for(; it != coords.end(); ++it) {
			const Coord3 c = m_tree->gridCoord((const float *)&(*it));
			if(added.find(c)) continue;
			added.insert(c);
			List<VertexP> * pl = m_tree->find((float *)&(*it));
			intersect(pl, *incident);
		}
		m_march.step();
	}
	
	m_active->finish();
}

void Sculptor::deselectPoints() 
{ 
	std::cout<<"grid count "<<m_tree->size();
	m_tree->calculateBBox();
	m_march.initialize(m_tree->boundingBox(), m_tree->gridSize());
	m_active->reset(); 
}

bool Sculptor::intersect(List<VertexP> * d, const Ray & ray)
{
	if(!d) return false;
	const int num = d->size();
	const int ndst = m_active->vertices->size();
	Vector3F pop;
	for(int i = 0; i < num; i++) {
		Vector3F & p = *(d->value(i).index->t1);
		float tt = ray.m_origin.dot(ray.m_dir) - p.dot(ray.m_dir);
		pop = ray.m_origin - ray.m_dir * tt;
		if(p.distanceTo(pop) < m_active->threshold) {
			int k = -tt / m_active->gridSize;
			m_active->vertices->insert(k, d->value(i));
			m_active->updateDepthRange(-tt);
		}
	}
	return m_active->vertices->size() > ndst;
}

C3Tree * Sculptor::allPoints() const { return m_tree; }

Sculptor::ActiveGroup * Sculptor::activePoints() const { return m_active; }

void Sculptor::pullPoints()
{
	if(m_active->numSelected() < 1) return;
	
	Ordered<int, VertexP> * vs = m_active->vertices;
	
	vs->elementBegin();
	while(!vs->elementEnd()) {
		VertexP * vert = vs->currentElement();
		Vector3F & pos = *(vert->index->t1);
		Vector3F p0(*(vert->index->t1));
		pos += m_active->meanNormal * 0.03f;
		
		m_tree->displace(*vert, p0);
		
		vs->nextElement();
	}
}

} // end namespace sdb