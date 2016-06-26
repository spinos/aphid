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

BccTetrahedralize::BccTetrahedralize()
{}

BccTetrahedralize::~BccTetrahedralize() 
{}

const char * BccTetrahedralize::titleStr() const
{ return "BCC Tetrahedralize"; }

bool BccTetrahedralize::createSamples()
{
	SuperformulaPoisson::createSamples();
	
	PoissonSequence<Disk> * supg = sampleGrid();
	
	const float supsize = supg->gridSize(); /// 4r
	m_mesher.setH(supsize);
	
	const float bsize = supsize * .5f; /// 2r
	m_pntSz = bsize * .0625f;
	BoundingBox cbx;
	
	supg->begin();
	while(!supg->end() ) {
		
		Vector3F center = supg->coordToCellCenter(supg->key() );
		m_mesher.addCell(center);
		
		supg->getCellBoundingBox(&cbx);
		
		Vector3F bcn = cbx.center();
		float bdx = cbx.distance(0) *.5f;
		float bdy = cbx.distance(1) *.5f;
		float bdz = cbx.distance(2) *.5f;
		
		for(int i=0; i<26; ++i) {
			Vector3F voffset(PoissonSequence<Disk>::TwentySixNeighborCoord[i][0],
							PoissonSequence<Disk>::TwentySixNeighborCoord[i][1],
							PoissonSequence<Disk>::TwentySixNeighborCoord[i][2]);
							
			center = bcn + Vector3F(bdx * voffset.x,
									bdy * voffset.y,
									bdz * voffset.z)
						+ voffset * bsize;
		
			m_mesher.addCell(center);
			
		}
		
		supg->next();
	}
	
	m_sampleBegin = m_mesher.finishGrid();
	
	m_mesher.setN(m_sampleBegin + sampleGrid()->elementCount() );
	
	const int Nv = m_mesher.N();
	
	
	extractSamplePos(&m_mesher.X()[m_sampleBegin]);
	
	std::cout<<"\n n tet b4 delauney "<<m_mesher.build();
	
	int i = m_sampleBegin;
	for(; i<Nv;++i) {
		if(!m_mesher.addPoint(i) ) {
			std::cout<<"\n [WARNING] add pnt break at v"<<i;
			break;
		}
		if(!m_mesher.checkConnectivity() ) {
			std::cout<<"\n [WARNING] check conn break at v"<<i;
			break;
		}
	}
	
	std::cout<<"\n n samples "<<m_sampleBegin
		<<"\n n total node "<<Nv
		<<"\n n tet "<<m_mesher.numTetrahedrons();
	std::cout.flush();
	return true;
}

void BccTetrahedralize::draw(aphid::GeoDrawer * dr)
{
	const int Nv = m_mesher.N();
	const int Nt = m_mesher.numTetrahedrons();
	const Vector3F * X = m_mesher.X();
	
	dr->m_markerProfile.apply();
	dr->setColor(0.f, 0.f, 0.f);
	int i;
#if 0
	dr->setColor(0.f, 0.f, .5f);
	for(i=0;i<m_sampleBegin;++i) {
		dr->cube(X[i], m_pntSz);
	}
#endif
	dr->setColor(0.f, .5f, 0.f);
	for(i=m_sampleBegin;i<Nv;++i) {
		dr->cube(X[i], m_pntSz);
	}
	
	dr->setColor(1.f, 0.f, 0.f);
	dr->cube(X[292], m_pntSz);
	dr->setColor(1.f, 1.f, 0.f);
	dr->cube(X[290], m_pntSz);
	dr->cube(X[39], m_pntSz);
	dr->cube(X[43], m_pntSz);
	
	dr->setColor(0.f, 1.f, 0.f);
	dr->cube(X[289], m_pntSz);
	dr->cube(X[39], m_pntSz);
	dr->cube(X[44], m_pntSz);
	dr->cube(X[37], m_pntSz);
	
	dr->setColor(0.f, 0.f, 0.f);
	float nmbSz = m_pntSz * 2.f;
	dr->drawNumber(292, X[292], nmbSz);
	dr->drawNumber(44, X[44], nmbSz);
	dr->drawNumber(289, X[289], nmbSz);
	dr->drawNumber(290, X[290], nmbSz);
	dr->drawNumber(39, X[39], nmbSz);
	dr->drawNumber(44, X[44], nmbSz);
	dr->drawNumber(43, X[43], nmbSz);
	dr->drawNumber(37, X[37], nmbSz);
	
	//dr->m_wireProfile.apply(); // slow
	dr->setColor(0.2f, 0.2f, 0.49f);
	
	glBegin(GL_LINES);
	Vector3F a, b, c, d;
	
	for(i=0; i<Nt; ++i) {
		const ITetrahedron * t = m_mesher.tetrahedron(i);
		
		if(t->index < 0) continue;
		
		a = X[t->iv0];
		b = X[t->iv1];
		c = X[t->iv2];
		d = X[t->iv3];
		
		glVertex3fv((const GLfloat *)&a);
		glVertex3fv((const GLfloat *)&b);
		
		glVertex3fv((const GLfloat *)&a);
		glVertex3fv((const GLfloat *)&c);
		
		glVertex3fv((const GLfloat *)&a);
		glVertex3fv((const GLfloat *)&d);
		
		glVertex3fv((const GLfloat *)&b);
		glVertex3fv((const GLfloat *)&c);
		
		glVertex3fv((const GLfloat *)&c);
		glVertex3fv((const GLfloat *)&d);
		
		glVertex3fv((const GLfloat *)&d);
		glVertex3fv((const GLfloat *)&b);
	}
	
	glEnd();
}

}
