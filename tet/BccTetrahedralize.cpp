/*
 *  BccTetrahedralize.cpp
 *  
 *
 *  Created by jian zhang on 6/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "BccTetrahedralize.h"
#include <iostream>

using namespace aphid;
namespace ttg {

BccTetrahedralize::BccTetrahedralize() :
m_X(NULL)
{}

BccTetrahedralize::~BccTetrahedralize() 
{
	if(m_X) delete[] m_X;
}

const char * BccTetrahedralize::titleStr() const
{ return "BCC Tetrahedralize"; }

bool BccTetrahedralize::createSamples()
{
	SuperformulaPoisson::createSamples();
	
	m_grid.clear();
	
	PoissonSequence<Disk> * supg = supportGrid();
	
	const float supsize = supg->gridSize();
	m_grid.setGridSize(supsize * 2.f);
	m_pntSz = supsize * .0625f;
	
	supg->begin();
	while(!supg->end() ) {
		
		Vector3F center = supg->coordToCellCenter(supg->key() );
		if(!m_grid.findCell(center) ) {
			BccNode * node15 = new BccNode;
			node15->key = 15;
			m_grid.insert((const float *)&center, node15 );
		}
		
		for(int i=0; i<6; ++i) {
			center = supg->coordToCellCenter(supg->key() )
						 + Vector3F( supsize * PoissonSequence<Disk>::TwentySixNeighborCoord[i][0],
										supsize * PoissonSequence<Disk>::TwentySixNeighborCoord[i][1],
										 supsize * PoissonSequence<Disk>::TwentySixNeighborCoord[i][2]);
		
			if(!m_grid.findCell(center) ) {
				BccNode * node15 = new BccNode;
				node15->key = 15;
				m_grid.insert((const float *)&center, node15 );
			}
		}
		
		supg->next();
	}
	m_grid.calculateBBox();
	std::cout<<"\n n bcc cell "<<m_grid.size();
	m_grid.buildNodes();
	m_N = m_grid.numNodes();
	m_supportBegin = m_N;
	m_N += supportGrid()->elementCount();
	m_sampleBegin = m_N;
	m_N += sampleGrid()->elementCount();
	std::cout<<"\n n total node "<<m_N;
	
	if(m_X) delete[] m_X;
	m_X = new Vector3F[m_N];
	m_grid.getNodePositions(m_X);
	
	extractSupportPos(&m_X[m_supportBegin]);
	extractSamplePos(&m_X[m_sampleBegin]);
	
	m_tets.clear();
	m_grid.buildTetrahedrons(m_tets);
	std::cout<<"\n n tet "<<m_tets.size();
	std::cout.flush();
	return true;
}

void BccTetrahedralize::draw(aphid::GeoDrawer * dr)
{
	dr->m_markerProfile.apply();
	dr->setColor(0.f, 0.f, 0.f);
	int i = 0;
	for(;i<m_supportBegin;++i) {
		dr->cube(m_X[i], m_pntSz);
	}
	dr->setColor(0.f, 0.f, .5f);
	for(i=m_supportBegin;i<m_sampleBegin;++i) {
		dr->cube(m_X[i], m_pntSz);
	}
	dr->setColor(0.f, .5f, 0.f);
	for(i=m_sampleBegin;i<m_N;++i) {
		dr->cube(m_X[i], m_pntSz);
	}
	
	dr->m_wireProfile.apply();
	dr->setColor(0.2f, 0.2f, 0.49f);
	
	glBegin(GL_TRIANGLES);
	Vector3F a, b, c, d;
	
	std::vector<ITetrahedron *>::const_iterator it = m_tets.begin();
	for(;it!= m_tets.end();++it) {
		const ITetrahedron * t = *it;
		
		a = m_X[t->iv0];
		b = m_X[t->iv1];
		c = m_X[t->iv2];
		d = m_X[t->iv3];
		
		glVertex3fv((const GLfloat *)&b);
		glVertex3fv((const GLfloat *)&c);
		glVertex3fv((const GLfloat *)&d);
		
		glVertex3fv((const GLfloat *)&a);
		glVertex3fv((const GLfloat *)&b);
		glVertex3fv((const GLfloat *)&d);
		
		glVertex3fv((const GLfloat *)&a);
		glVertex3fv((const GLfloat *)&c);
		glVertex3fv((const GLfloat *)&b);
		
		glVertex3fv((const GLfloat *)&a);
		glVertex3fv((const GLfloat *)&d);
		glVertex3fv((const GLfloat *)&c);
	}
	
	glEnd();
}

}
