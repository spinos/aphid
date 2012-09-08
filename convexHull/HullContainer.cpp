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

int HullContainer::getNumVertex() const
{
	return m_vertices.size();
}
	
int HullContainer::getNumFace() const
{
	return m_faces.size();
}

void HullContainer::addVertex(Vertex *p)
{
	p->setIndex(getNumVertex());
	m_vertices.push_back(p);
}

void HullContainer::addFacet(Facet *f)
{
	f->setIndex(getNumFace());
	m_faces.push_back(f);
	printf("add face %d\n", f->getIndex());
}

void HullContainer::removeFaces()
{
	//printf("remove face\n");
	//printf("b4\n");
	std::vector<Facet *>::iterator it;
	std::vector<Facet *>::iterator rest;
	//for(it = m_faces.begin(); it < m_faces.end(); it++ )
	//	printf("%d ", (*it)->getIndex());

	int i = 0;
	for(it = m_faces.begin(); it < m_faces.end(); it++)
	{ 
		if((*it)->getIndex() < 0)
		{
			for(rest = m_faces.begin() + i; rest < m_faces.end(); rest++)
			{
				(*rest)->setIndex((*rest)->getIndex() - 1);
			}
			m_faces.erase(m_faces.begin() + i);
			it--;
			i--;
		}
		i++;
	}
	
	//printf("\naft\n");
	//for(it = m_faces.begin(); it < m_faces.end(); it++ )
	//	printf("%d ", (*it)->getIndex());

	//printf("\n");
	
}

Facet HullContainer::getFacet(int idx) const
{
	return *m_faces[idx];
}

Vertex HullContainer::getVertex(int idx) const
{
	return *m_vertices[idx];
}

Vertex *HullContainer::vertex(int idx)
{
	return m_vertices[idx];
}

void HullContainer::initHull()
{
	fDrawer = new ShapeDrawer();
	int nv = 603;
	for(int i = 0; i < nv; i++) 
	{
		Vertex * v = new Vertex;
		//float r = ((float)(rand() % 4091)) / 4091.f * 13.f + 3.f;
		//float phi = ((float)(rand() % 5391)) / 5391.f * 2.f * 3.14f;
		//float theta = ((float)(rand() % 4331)) / 4331.f * 3.14f;
		//v->x = sin(theta) * cos(phi) * r + 16.f;
		//v->y = sin(theta) * sin(phi) * r + 16.f;
		//v->z = cos(theta) * r + 16.f;
		v->x = ((float)(rand() % 8091)) / 8092.f * 32;
		v->y = ((float)(rand() % 8391)) / 8392.f * 32;
		v->z = ((float)(rand() % 8331)) / 8332.f * 32;
		addVertex(v);
		v->setData((char*)new ConflictGraph(0));
	}
	beginHull();
}

void HullContainer::killHull()
{
	m_vertices.clear();
	m_faces.clear();
}

void HullContainer::beginHull()
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
	
	for(int i = 4; i < getNumVertex(); i++)
	{
		Vertex *q = vertex(i);
		if(searchVisibleFaces(q))
		{
			m_currentVertexId = i;
			if(searchHorizons())
			{
				spawn(q);
				finishStep(q);
			}
			else 
			{
				printf("horizon failed at v %d", i);
				break;
			}
		}
	}
}

char HullContainer::searchVisibleFaces(Vertex *v)
{
	visibleFaces.clear();
	((ConflictGraph *)v->getData())->getFaces(visibleFaces);
	if(visibleFaces.size() < 1) return 0;
	printf("%d faces are visible\n", (int)visibleFaces.size());
	return 1;
}

char HullContainer::searchHorizons()
{
	std::vector<Facet *>::iterator it;
	//for(it = m_faces.begin(); it < m_faces.end(); it++ )
	//	(*it)->setMarked(0);
	
	for (it = visibleFaces.begin(); it < visibleFaces.end(); it++) 
	{ 
		printf("mark faces %d\n", (*it)->getIndex());
		(*it)->setMarked(1);
	}
	
	std::vector<Edge *>horizons;

	for(it = visibleFaces.begin(); it < visibleFaces.end(); it++) 
	{ 
		printf("get horizon from face %d\n", (*it)->getIndex());
		(*it)->getEdgeOnHorizon(horizons);
	}
	
	if(horizons.size() < 1)
	{
		printf("no horizon\n");
		return 0;
	}
	
	printf("%d horizon edges\n", (int)horizons.size());
	
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
	m_numHorizon = 0;
	while(cur && i < (int)horizons.size())
	{
		Vertex * a = cur->v0();
		Vertex * b = cur->v1();
		printf("%f %f %f - %f %f %f\n", a->x, a->y, a->z, b->x, b->y, b->z);
		if( b->getIndex() == m_horizon->v0()->getIndex()) {
			cur->disconnect();
			loop = 1;
			printf("found loop\n");
			break;
		}
		cur = (Edge *)cur->getNext();
		i++;
		m_numHorizon++;
	}
	
	horizons.clear();
	
	if(!loop) 
	{
		printf("no loop\n");
		return 0;
	}
	
	return 1;
}

void HullContainer::spawn(Vertex *v)
{
	Facet *last = 0;
	Facet *first = 0;
	Vertex * end = 0;
	
	Edge *cur = m_horizon;
	while(cur)
	{
		Edge *e = cur;
		Vertex * a = e->v0();
		Vertex * b = e->v1();
		Facet *wall = (Facet *)(e->getFace());
		Facet *yard = (Facet *)(e->getTwin()->getFace());
		Vertex * c = wall->thirdVertex(a, b);
		Facet *f = new Facet(v, a, b, c);
		f->setData((char*)new ConflictGraph(1));
		
		addFacet(f);
		
		f->connectTo(wall, a, b);
		
		if(!first) first = f;
		
		if(last) f->connectTo(last, v, a);
		
		last = f;
		end = b;
		
		addConflict(f, wall, yard);
		
		cur = (Edge *)cur->getNext();
	}
	last->connectTo(first, v, end);
}

void HullContainer::finishStep(Vertex *v)
{
	v->setVisibility(0);
	
	for(int i = 0; i < (int)visibleFaces.size(); i++)
	{
		Facet *f = visibleFaces[i];
		//((ConflictGraph *)f->getData())->clear();
		//removeConflict(f);
		printf("\n rm face %d\n", f->getIndex());

		f->setIndex(-1);
	}
	removeFaces();
}

void HullContainer::renderWorld()
{
	fDrawer->box(32, 32, 32);
	fDrawer->setGrey(.8f);
	fDrawer->beginPoint();
	for(int i = 0; i < getNumVertex(); i++) 
	{
		const Vertex p = getVertex(i);
		fDrawer->aVertex(p.x, p.y, p.z);
	}
	fDrawer->end();
	
	fDrawer->beginWireTriangle();
	
	std::vector<Facet *>::iterator it;
	for(it = m_faces.begin(); it < m_faces.end(); it++ )
	{
		const Facet *f = *it;

		fDrawer->setColor(0.f, .75f, 0.f);
		Vertex p = f->getVertex(0);
		fDrawer->aVertex(p.x, p.y, p.z);
		p = f->getVertex(1);
		fDrawer->aVertex(p.x, p.y, p.z);
		p = f->getVertex(2);
		fDrawer->aVertex(p.x, p.y, p.z);
	}
	
	fDrawer->end();
	
	fDrawer->setGrey(1.f);
	
	fDrawer->beginSolidTriangle();
	for(it = visibleFaces.begin(); it < visibleFaces.end(); it++ )
	{
		const Facet *f = *it;
		Vertex p = f->getVertex(0);
		fDrawer->aVertex(p.x, p.y, p.z);
		p = f->getVertex(1);
		fDrawer->aVertex(p.x, p.y, p.z);
		p = f->getVertex(2);
		fDrawer->aVertex(p.x, p.y, p.z);
	}
	fDrawer->end();
	
	fDrawer->beginLine();
	
	for(it = m_faces.begin(); it < m_faces.end(); it++ )
	{
		const Facet *f = *it;
		Vector3F c = f->getCentroid();
		Vector3F nor = f->getNormal();
		fDrawer->aVertex(c.x, c.y, c.z);
		fDrawer->aVertex(c.x + nor.x, c.y + nor.y, c.z + nor.z);
	}
	
	fDrawer->end();
	
	fDrawer->setColor(1.f, 0.f, 0.f);
	
	const Vertex pv = getVertex(m_currentVertexId);
	fDrawer->solidCube(pv.x, pv.y, pv.z, 0.5f);
	
	fDrawer->beginLine();
	
	Edge *cur = m_horizon;
	int i = 0;
	while(cur && i <= m_numHorizon) 
	{
		const Vertex a = cur->getV0();
		const Vertex b = cur->getV1();
		fDrawer->aVertex(a.x, a.y, a.z);
		fDrawer->aVertex(b.x, b.y, b.z);
		cur = (Edge *)cur->getNext();
		i++;
	}
	
	fDrawer->end();
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
		if (f->isVertexAbove(*v) && v->isVisible()) addConflict(f, v);
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

