/*
 *  ConvexHullGen.cpp
 *  qtbullet
 *
 *  Created by jian zhang on 7/13/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>
#include "ConvexHullGen.h"
#include <topo/Vertex.h>
#include <topo/Facet.h>
#include <topo/Edge.h>
#include <topo/GraphArch.h>
#include <topo/ConflictGraph.h>
#include <geom/ATriangleMesh.h>
#include <cmath>

namespace aphid {

ConvexHullGen::ConvexHullGen() :
m_horizon(0)
{}

ConvexHullGen::~ConvexHullGen() 
{
    if(m_horizon) {
        delete m_horizon;
    }
	visibleFaces.clear();
	
	std::vector<ConflictGraph *>::iterator it = m_conflg.begin();
	for(;it!=m_conflg.end();++it) {
		//delete *it;
	}
	m_conflg.clear();
	std::vector<GraphArch *>::iterator it1 = m_arch.begin();
	for(;it1!=m_arch.end();++it1) {
		//delete *it1;
	}
	m_arch.clear();
}

void ConvexHullGen::addSample(const Vector3F & p)
{
	Vertex * vet = new Vertex;
	vet->m_v = new Vector3F;
	*vet->m_v = p;
	addVertex(vet);
	ConflictGraph * cfg = new ConflictGraph(0);
	vet->setData((char*)cfg);
	m_conflg.push_back(cfg);
}

void ConvexHullGen::processHull()
{
	Vertex *a = vertex(0);
	Vertex *b = vertex(1);
	Vertex *c = vertex(2);
	Vertex *d = vertex(3);
	
	Facet *f1 = new Facet(a, b, c, d->m_v);
	Facet *f2 = new Facet(a, c, d, b->m_v);
	Facet *f3 = new Facet(a, d, b, c->m_v);
	Facet *f4 = new Facet(b, d, c, a->m_v);
	
	ConflictGraph * cfg1 = new ConflictGraph(1);
	ConflictGraph * cfg2 = new ConflictGraph(1);
	ConflictGraph * cfg3 = new ConflictGraph(1);
	ConflictGraph * cfg4 = new ConflictGraph(1);
	
	f1->setData((char*)cfg1);
	f2->setData((char*)cfg2);
	f3->setData((char*)cfg3);
	f4->setData((char*)cfg4);
	
	m_conflg.push_back(cfg1);
	m_conflg.push_back(cfg2);
	m_conflg.push_back(cfg3);
	m_conflg.push_back(cfg4);
	
	addFacet(f1);
	addFacet(f2);
	addFacet(f3);
	addFacet(f4);
	
	f1->connectTo(f2, a, c);
	f1->connectTo(f3, a, b);
	f1->connectTo(f4, b, c);
	f2->connectTo(f3, a, d);
	f2->connectTo(f4, c, d);
	f3->connectTo(f4, b, d);
	
	for (int i=4; i < getNumVertex(); i++) {
	 Vertex *v = vertex(i);
	 if (f1->isVertexAbove(*v)) addConflict(f1, v);
	 if (f2->isVertexAbove(*v)) addConflict(f2, v);
	 if (f3->isVertexAbove(*v)) addConflict(f3, v);
	 if (f4->isVertexAbove(*v)) addConflict(f4, v);
	}
	
	a->setVisibility(0);
	b->setVisibility(0);
	c->setVisibility(0);
	d->setVisibility(0);
	
	m_currentVertexId = 4;
	int i;
	for(i = 4; i < getNumVertex(); i++)
	{
		Vertex *q = vertex(i);
		if(searchVisibleFaces(q))
		{
			m_currentVertexId = i;
			if(searchHorizons())
			{
				if(!spawn(q)) {
#ifndef NDEBUG
					printf("spawn failed at v %d", i);
#endif
					break;
				}
				if(!finishStep(q))
				{
#ifndef NDEBUG
					printf("spawn failed at v %d", i);
#endif
					break;
				}
			}
		}
	}
#ifndef NDEBUG	
	if(i == getNumVertex()) {
		printf("well done!");
	}
#endif
}

char ConvexHullGen::searchVisibleFaces(Vertex *v)
{
	visibleFaces.clear();
	((ConflictGraph *)v->getData())->getFaces(visibleFaces);
	if(visibleFaces.size() < 1) return 0;
#ifndef NDEBUG
	printf("%d faces are visible\n", (int)visibleFaces.size());
#endif
	return 1;
}

char ConvexHullGen::searchHorizons()
{
	std::vector<Facet *>::iterator it = faces().begin();
	for(; it < faces().end(); it++ ) {
		(*it)->setMarked(0);
	}
	
	for (it = visibleFaces.begin(); it < visibleFaces.end(); it++) 
	{ 
#ifndef NDEBUG
		printf("mark faces %d\n", (*it)->getIndex());
#endif
		(*it)->setMarked(1);
	}
	
	std::vector<Edge *>horizons;

	for(it = visibleFaces.begin(); it < visibleFaces.end(); it++) 
	{
#ifndef NDEBUG
		printf("get horizon from face %d\n", (*it)->getIndex());
#endif
		if(!(*it)->getEdgeOnHorizon(horizons))
		{
#ifndef NDEBUG
			printf("face not connected\n");
#endif
			return 0;
		}
	}
	
	if(horizons.size() < 3)
	{
#ifndef NDEBUG
		printf("horizon less than 3\n");
#endif
		return 0;
	}
#ifndef NDEBUG	
	printf("%d horizon edges\n", (int)horizons.size());
#endif	
	Edge *cur = horizons.at(0);
	for (int j=1; j<(int)horizons.size(); j++) 
	{
		Edge *e;
		for (int i=1; i<(int)horizons.size(); i++) 
		{ 
			e = horizons.at(i);
			if(cur->canBeConnectedTo(e)) {
				cur->connect(e);
				cur = e;
				break;
			}	
		}
	}
	
	cur = horizons.at(0);
	m_horizon = cur;
	int i = 0;
	char loop = 0;
	m_numHorizon = 1;
	while(cur && i < (int)horizons.size())
	{
		//Vertex * a = cur->v0();
		Vertex * b = cur->v1();
		//printf("%f %f %f - %f %f %f\n", a->x, a->y, a->z, b->x, b->y, b->z);
		if( b->getIndex() == m_horizon->v0()->getIndex()) {
			cur->disconnect();
			loop = 1;
#ifndef NDEBUG
			printf("found loop\n");
#endif
			break;
		}
		cur = (Edge *)cur->getNext();
		i++;
		m_numHorizon++;
	}
	
	int numE = (int)horizons.size();
	horizons.clear();
	
	if(!loop) 
	{
#ifndef NDEBUG
		printf("no loop\n");
#endif
		return 0;
	}
#ifndef NDEBUG	
	printf("num horizon %d\n", m_numHorizon);
#endif
	if(m_numHorizon < 3 || m_numHorizon != numE)
	{
#ifndef NDEBUG
		printf("unexpected horizon loop\n");
#endif
		return 0;
	}
	
	return 1;
}

char ConvexHullGen::spawn(Vertex *v)
{
	Edge *cur = m_horizon;
	Vector3F horizonCen(0.f, 0.f, 0.f);
	for(int i = 0; i < (int)visibleFaces.size(); i++) {
		horizonCen += visibleFaces.at(i)->getCentroid();
	}
	horizonCen /= (float)visibleFaces.size();
	
	Facet *last = 0;
	Facet *first = 0;
	Vertex * end = 0;
	
	cur = m_horizon;
	while(cur)
	{
		Edge *e = cur;
		Vertex * a = e->v0();
		Vertex * b = e->v1();
		Facet *wall = (Facet *)(e->getFace());
		if(wall->getIndex() < 0 || wall->isMarked())
		{
#ifndef NDEBUG
			printf("face %d is not wall\n", wall->getIndex());
#endif
			return 0;
		}
		Facet *yard = (Facet *)(e->getTwin()->getFace());
		if(yard->getIndex() < 0)
		{
#ifndef NDEBUG
			printf("face %d is not yard\n", yard->getIndex());
#endif
			return 0;
		}
		
		//Vertex * c = yard->thirdVertex(a, b);
		Facet *f = new Facet(v, a, b, &horizonCen);
		
		ConflictGraph * cfg = new ConflictGraph(1);
		f->setData((char*)cfg);
		
		m_conflg.push_back(cfg);
		
		addFacet(f);
		
		f->connectTo(wall, a, b);
		
		if(!first) first = f;
		
		if(last) {
			if(!f->connectTo(last, v, a))
				return 0;
		}
		
		last = f;
		end = b;
		
		addConflict(f, wall, yard);
		
		cur = (Edge *)cur->getNext();
	}
	return last->connectTo(first, v, end);
}

char ConvexHullGen::finishStep(Vertex *v)
{
	v->setVisibility(0);
	
	for(int i = 0; i < (int)visibleFaces.size(); i++)
	{
		Facet *f = visibleFaces[i];
		removeConflict(f);
		
#ifndef NDEBUG
		printf(" rm face %d\n", f->getIndex());
#endif

		f->setIndex(-1);
	}
	removeFaces();
	
	std::vector<Facet *>::iterator it = faces().begin();
	for(; it < faces().end(); it++ )
	{
		if(!(*it)->isClosed()) {
#ifndef NDEBUG
			printf("face %d is not closed\n", (*it)->getIndex());
#endif
			return 0;
		}
	}
	return 1;
}

void ConvexHullGen::addConflict(Facet *f, Vertex *v)
{
	GraphArch *arc = new GraphArch(f, v);
      ((ConflictGraph *)f->getData())->add(arc);
      ((ConflictGraph *)v->getData())->add(arc);
	m_arch.push_back(arc);
}

void ConvexHullGen::addConflict(Facet *f, Facet *f1, Facet *f2)
{
	std::vector<Vertex *> f1Visible;
	std::vector<Vertex *> f2Visible;
	((ConflictGraph *)f1->getData())->getVertices(f1Visible);
	((ConflictGraph *)f2->getData())->getVertices(f2Visible);
	
	Vertex *v1;
	Vertex *v2;
	int i1 = 0, i2 = 0;
	std::vector<Vertex *> visible;
	while(i1 < (int)f1Visible.size() || i2 < (int)f2Visible.size())
	{
		if(i1 < (int)f1Visible.size() && i2 < (int)f2Visible.size())
		{
			v1 = (Vertex *)f1Visible.at(i1);
			v2 = (Vertex *)f2Visible.at(i2);
			if(v1->getIndex() == v2->getIndex())
			{
				visible.push_back(v1);
				i1++;
				i2++;
			}
			else if(v1->getIndex() > v2->getIndex())
			{
				visible.push_back(v1);
				i1++;
			}
			else
			{
				visible.push_back(v2);
				i2++;
			}
		}
		else if(i1 < (int)f1Visible.size())
		{
			v1 = (Vertex *)f1Visible.at(i1);
			visible.push_back(v1);
			i1++;
		}
		else
		{
			v2 = (Vertex *)f2Visible.at(i2);
			visible.push_back(v2);
			i2++;
		}
	}
	
	for(int i=(int)visible.size() - 1; i >= 0; i--) {
		Vertex *v = visible.at(i);
		if (f->isVertexAbove(*v)) {
			addConflict(f, v);
		}
	}
}

void ConvexHullGen::removeConflict(Facet *f)
{
	Vertex *conflictedV = new Vertex;
	((ConflictGraph *)f->getData())->getVertices(conflictedV);
	conflictedV = (Vertex *)conflictedV->getNext();
	while(conflictedV) 
	{
		((ConflictGraph *)conflictedV->getData())->removeFace(f);
		conflictedV = (Vertex *)conflictedV->getNext();
	}
	delete conflictedV;
}

Vector3F ConvexHullGen::getCenter() const
{
	Vector3F c(0.f, 0.f, 0.f);
	std::vector<Facet *>::const_iterator it = visibleFaces.begin();
	for(;it!=visibleFaces.end();++it) {
		c += (*it)->getCentroid();
	}
	c /= (float)visibleFaces.size();
	return c;
}

void ConvexHullGen::checkFaceNormal(Vector3F* pos, Vector3F* nml,
			const Vector3F & vref) const
{
	Vector3F va = pos[1] - pos[0];
	Vector3F vb = pos[2] - pos[0];
	Vector3F vn = va.cross(vb);
	vn.normalize();
	if(vn.dot(vref) < 0.f) { std::cout<<" revsd";
		vn.reverse();
		va = pos[1];
		pos[1] = pos[2];
		pos[2] = va;
	}
	for(int j=0;j<3;++j) {
		nml[j] = vn;
	}
}

void ConvexHullGen::extractMesh(ATriangleMesh * msh)
{
	const int nt = getNumFace();
	const int nv = nt * 3;
	msh->create(nv, nt);
	
	unsigned * indDst = msh->indices();
    Vector3F * pntDst = msh->points();
    Vector3F * nmlDst = msh->vertexNormals();
	
	const Vector3F hc = getCenter();
	
	for(int i=0;i<nt;++i) {
		const Facet & fc = getFacet(i);
		Vector3F dv = fc.getCentroid() - hc;
		
		for(int j=0;j<3;++j) {
			const Vertex * vj = fc.vertex(j);
			pntDst[i*3 + j] = *(vj->m_v);
			indDst[i*3 + j] = i*3 + j;
		}
		checkFaceNormal(&pntDst[i*3], &nmlDst[i*3], dv);
		
	}
}

void ConvexHullGen::extractMesh(Vector3F* pos, Vector3F* nml, unsigned* inds, int offset)
{
	const Vector3F hc = getCenter();
	
	const int nt = getNumFace();
	for(int i=0;i<nt;++i) {
		const Facet & fc = getFacet(i);
		
		Vector3F dv = fc.getCentroid() - hc;
		
		for(int j=0;j<3;++j) {
			const Vertex * vj = fc.vertex(j);
			pos[i*3 + j] = hc + (*(vj->m_v) - hc) * 1.29f;
			pos[i*3 + j].y = vj->m_v->y;
			inds[i*3 + j] = i*3 + j + offset;
		}
		checkFaceNormal(&pos[i*3], &nml[i*3], dv);
		
	}
}

}
