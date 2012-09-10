/*
 *  HullContainer.cpp
 *  qtbullet
 *
 *  Created by jian zhang on 7/13/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>
#include "HullContainer.h"
#include <ConflictGraph.h>
#include <cmath>

HullContainer::HullContainer() {}
HullContainer::~HullContainer() {}

void HullContainer::initHull()
{
	int nv = 9999;
	for(int i = 0; i < nv; i++) 
	{
		Vertex * v = new Vertex;
		float r = ((float)(rand() % 24091)) / 24091.f * 10.f + 12.f;
		float phi = ((float)(rand() % 25391)) / 25391.f * 2.f * 3.14f;
		float theta = ((float)(rand() % 24331)) / 24331.f * 3.14f;
		v->x = sin(theta) * cos(phi) * r + 16.f;
		v->y = sin(theta) * sin(phi) * r + 16.f;
		v->z = cos(theta) * r + 16.f;
		//v->x = ((float)(rand() % 218091)) / 8092.f;
		//v->y = ((float)(rand() % 308391)) / 8392.f;
		//v->z = ((float)(rand() % 298331)) / 8332.f;
		addVertex(v);
		v->setData((char*)new ConflictGraph(0));
	}
	processHull();
}

void HullContainer::processHull()
{
	Vertex *a = vertex(0);
	Vertex *b = vertex(1);
	Vertex *c = vertex(2);
	Vertex *d = vertex(3);
	
	Facet *f1 = new Facet(a, b, c, d);
	Facet *f2 = new Facet(a, c, d, b);
	Facet *f3 = new Facet(a, b, d, c);
	Facet *f4 = new Facet(b, c, d, a);
	
	f1->setData((char*)new ConflictGraph(1));
	f2->setData((char*)new ConflictGraph(1));
	f3->setData((char*)new ConflictGraph(1));
	f4->setData((char*)new ConflictGraph(1));
	
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
	if(i == getNumVertex())
		printf("well done!");
#endif
}

char HullContainer::searchVisibleFaces(Vertex *v)
{
	visibleFaces.clear();
	((ConflictGraph *)v->getData())->getFaces(visibleFaces);
	if(visibleFaces.size() < 1) return 0;
#ifndef NDEBUG
	printf("%d faces are visible\n", (int)visibleFaces.size());
#endif
	return 1;
}

char HullContainer::searchHorizons()
{
	std::vector<Facet *>::iterator it;
	for(it = m_faces.begin(); it < m_faces.end(); it++ )
		(*it)->setMarked(0);
	
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

char HullContainer::spawn(Vertex *v)
{
	Edge *cur = m_horizon;
	Vector3F horizonCen(0.f, 0.f, 0.f);
	for(int i = 0; i < (int)visibleFaces.size(); i++)
	{
		
		horizonCen += visibleFaces.at(i)->getCentroid() - visibleFaces.at(i)->getNormal();
		
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
		f->setData((char*)new ConflictGraph(1));
		
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

char HullContainer::finishStep(Vertex *v)
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
	
	std::vector<Facet *>::iterator it;
	for(it = m_faces.begin(); it < m_faces.end(); it++ )
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

void HullContainer::renderWorld(ShapeDrawer * drawer)
{
	std::vector<Facet *>::iterator it;
	
	drawer->setGrey(1.f);
	
	drawer->beginSolidTriangle();
	for(it = visibleFaces.begin(); it < visibleFaces.end(); it++ )
	{
		const Facet *f = *it;
		Vertex p = f->getVertex(0);
		drawer->aVertex(p.x, p.y, p.z);
		p = f->getVertex(1);
		drawer->aVertex(p.x, p.y, p.z);
		p = f->getVertex(2);
		drawer->aVertex(p.x, p.y, p.z);
	}
	drawer->end();
	
	drawer->setColor(1.f, 0.f, 0.f);
	
	const Vertex pv = getVertex(m_currentVertexId);
	drawer->solidCube(pv.x, pv.y, pv.z, 0.5f);
	
	drawer->beginLine();
	
	Edge *cur = m_horizon;
	int i = 0;
	while(cur && i <= m_numHorizon) 
	{
		const Vertex a = cur->getV0();
		const Vertex b = cur->getV1();
		drawer->aVertex(a.x, a.y, a.z);
		drawer->aVertex(b.x, b.y, b.z);
		cur = (Edge *)cur->getNext();
		i++;
	}
	
	drawer->end();
}

void HullContainer::addConflict(Facet *f, Vertex *v)
{
	GraphArch *arc = new GraphArch(f, v);
      ((ConflictGraph *)f->getData())->add(arc);
      ((ConflictGraph *)v->getData())->add(arc);
}

void HullContainer::addConflict(Facet *f, Facet *f1, Facet *f2)
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
	
	printf(" \n");

	for(int i=(int)visible.size() - 1; i >= 0; i--) 
	{
		Vertex *v = visible.at(i);
		if (f->isVertexAbove(*v)) addConflict(f, v);
	}
}

void HullContainer::removeConflict(Facet *f)
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

