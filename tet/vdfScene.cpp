/*
 *  vdfScene.cpp
 *  foo
 *
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "vdfScene.h"
#include <NTreeDrawer.h>
#include "FieldTriangulation.h"
#include <iostream>

using namespace aphid;
namespace ttg {

vdfScene::vdfScene(const std::string & filename) 
{
	m_fileName = filename;
	m_msh = new FieldTriangulation;
}

vdfScene::~vdfScene() 
{ delete m_msh; }
	
const char * vdfScene::titleStr() const
{ return "Geom Distance Field"; }

bool vdfScene::viewPerspective() const
{ return true; }

bool vdfScene::init()
{
/// no grid
	m_container.readTree(m_fileName, 0);
	
	BoundingBox tb = m_container.tree()->getBBox();
/// larger grid size less faces
	const float gz = tb.getLongestDistance() * .53f;
	std::cout<<"\n gz "<<gz;
	const Vector3F cent = tb.center();
	tb.setMin(cent.x - gz, cent.y - gz, cent.z - gz );
	tb.setMax(cent.x + gz, cent.y + gz, cent.z + gz );
	setColorScale(1.f / gz);
	setNodeDrawSize(gz * GDT_FAC_ONEOVER128 );
	
	m_msh->fillBox(tb, gz);
	
	m_distFunc.addTree(m_container.tree() );
	m_distFunc.setDomainDistanceRange(gz * GDT_FAC_ONEOVER16 );
	m_distFunc.setShellThickness(0.f);
	m_distFunc.setSplatRadius(gz * GDT_FAC_ONEOVER32);
	
	m_msh->marchFrontBuild<BDistanceFunction>(&m_distFunc, 5);
	//m_msh->triangulateFront();
	
	std::cout.flush();
	return true;
}

void vdfScene::draw(GeoDrawer * dr)
{
	dr->setColor(.15f, .15f, .15f);
	dr->boundingBox(m_msh->grid()->boundingBox() );
		
#define SHO_TREE 0
#define SHO_CELL 1
#define SHO_FRONT_CELL 0
#define SHO_NODE 1
#define SHO_EDGE 0
#define SHO_ERR 0
#define SHO_FRONT 0
#define SHO_FRONT_WIRE 0

#if SHO_NODE
	drawNodes(m_msh, dr, true);
#endif

#if SHO_TREE
	drawTree(dr);
#endif

#if SHO_CELL
	dr->m_wireProfile.apply();
	drawGridCell<AdaptiveBccGrid3>(m_msh->grid(), dr, 4, 5);
#endif

#if SHO_EDGE	
	dr->m_wireProfile.apply();
	//drawFrontEdges(m_msh, dr);
	dr->setColor(0.f, 0.f, .5f);
	drawEdges(m_msh, dr, true);
#endif

#if SHO_FRONT	
	dr->m_surfaceProfile.apply();
	dr->setColor(0.f, .4f, .5f);
	drawFront<FieldTriangulation >(m_msh);
#endif
		
#if SHO_FRONT_WIRE	
	dr->m_wireProfile.apply();
	dr->setColor(0.1f, .1f, .1f);
	drawFrontWire<FieldTriangulation >(m_msh);
#endif

#if SHO_FRONT_CELL
	dr->m_wireProfile.apply();
	dr->setColor(0.25f, 0.25f, 0.45f);
	drawGridCell<AdaptiveBccGrid3>(m_msh->grid(), dr, 2, 2);
	dr->setColor(0.85f, 0.55f, 0.f);
	drawInteriorGridCell<AdaptiveBccGrid3>(m_msh->grid(), 2, dr);
	dr->setColor(0.f, 0.75f, 0.55f);
	drawFrontGridCell<AdaptiveBccGrid3>(m_msh->grid(), 2, dr);
#endif
	
}

void vdfScene::drawTree(aphid::GeoDrawer * dr)
{
#if 0
	if(!m_container.tree() ) return; 
	
	dr->m_wireProfile.apply();
	dr->setColor(.15f, .25f, .35f);
	dr->boundingBox(m_container.tree()->getBBox() );
	
	NTreeDrawer tdr;
	tdr.drawTree<cvx::Triangle>(m_container.tree() );
#endif

	if(!m_container.source() ) return;
	
	dr->m_surfaceProfile.apply();
	//dr->m_wireProfile.apply();
	dr->setColor(.8f, .8f, .8f);
		
	const sdb::VectorArray<cvx::Triangle> * src = m_container.source();
	const int n = src->size();
	glBegin(GL_TRIANGLES);
	int i=0;
	for(;i<n;++i) {
		const cvx::Triangle * t = src->get(i);
		
		glNormal3fv((GLfloat *)&t->N(0) );
		glVertex3fv((GLfloat *)t->p(0) );
		
		glNormal3fv((GLfloat *)&t->N(1) );
		glVertex3fv((GLfloat *)t->p(1) );
		
		glNormal3fv((GLfloat *)&t->N(2) );
		glVertex3fv((GLfloat *)t->p(2) );
	}
	glEnd();
}
}