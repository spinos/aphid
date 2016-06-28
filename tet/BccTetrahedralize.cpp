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
	
	const float supsize = supg->gridSize(); /// 2r
	m_mesher.setH(supsize);
	
	const float bsize = supsize * .5f; /// r
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
	
/// distort grid by red 
	supg->begin();
	while(!supg->end() ) {
	
		std::vector<Vector3F> smps;
		extractSamplePosIn(smps, supg->value() );
		
		Vector3F center = supg->coordToCellCenter(supg->key() );
		m_mesher.moveRedNodeInCell(center, smps);
		
		smps.clear();
		
		supg->next();
	}

/// smooth grid by blue
	supg->begin();
	while(!supg->end() ) {
		
		Vector3F center = supg->coordToCellCenter(supg->key() );
		m_mesher.smoothBlueNodeInCell(center);
		
		supg->next();
	}
	
#if 1		
	bool topoChanged;
	int i = m_sampleBegin;
	for(; i<Nv;++i) {
		if(!m_mesher.addPoint(i, topoChanged) ) {
			std::cout<<"\n [WARNING] add pnt break at v"<<i;
			break;
		}
		if(topoChanged) {
			if(!m_mesher.checkConnectivity() ) {
			std::cout<<"\n [WARNING] check conn break at v"<<i;
			break;
			}
			// std::cout<<"\n [INFO] passed topology check";
		}
	}
#endif	
	std::cout<<"\n n samples "<<m_sampleBegin
		<<"\n n total node "<<Nv
		<<"\n n tet "<<m_mesher.numTetrahedrons()
		<<"\n n front face "<<m_mesher.buildFrontFaces();
	std::cout.flush();
	return true;
}

void BccTetrahedralize::draw(aphid::GeoDrawer * dr)
{
	const int Nv = m_mesher.N();
	const int Nt = m_mesher.numTetrahedrons();
	const Vector3F * X = m_mesher.X();
	const int * prop = m_mesher.prop();
	
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
	
	Vector3F a, b, c, d;
	sdb::Array<sdb::Coord3, IFace > * fronts = m_mesher.frontFaces();
	dr->setColor(0.3f, 0.59f, 0.4f);
	
	glBegin(GL_TRIANGLES);
	fronts->begin();
	while(!fronts->end() ) {
		a = X[fronts->key().x];
		b = X[fronts->key().y];
		c = X[fronts->key().z];
		
		glVertex3fv((const float *)&a);
		glVertex3fv((const float *)&b);
		glVertex3fv((const float *)&c);
		
		fronts->next();
	}
	glEnd();
	
#if 0
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
	dr->drawNumber(258, X[258], nmbSz);
	dr->drawNumber(589, X[589], nmbSz);
	dr->drawNumber(276, X[276], nmbSz);
	dr->drawNumber(258, X[258], nmbSz);
	dr->drawNumber(260, X[260], nmbSz);
	dr->drawNumber(547, X[547], nmbSz);
	dr->drawNumber(373, X[373], nmbSz);
#endif
	
	//dr->m_wireProfile.apply(); // slow
	dr->setColor(0.2f, 0.2f, 0.49f);
	drawFrontEdges();
	
}

void BccTetrahedralize::drawFrontEdges()
{
	Vector3F a, b, c, d;
	int ra, rb, rc, rd;
	const int Nt = m_mesher.numTetrahedrons();
	const Vector3F * X = m_mesher.X();
	const int * prop = m_mesher.prop();
	
	glBegin(GL_LINES);
	for(int i=0; i<Nt; ++i) {
		const ITetrahedron * t = m_mesher.frontTetrahedron(i, 3);
		if(!t) continue;
		
		a = X[t->iv0];
		b = X[t->iv1];
		c = X[t->iv2];
		d = X[t->iv3];
		
		ra = prop[t->iv0];
		rb = prop[t->iv1];
		rc = prop[t->iv2];
		rd = prop[t->iv3];
		
		//dr->tetrahedronWire(a, b, c, d);
		if(ra > -1 && rb > -1) {
			glVertex3fv((const float *)&a);
			glVertex3fv((const float *)&b);
		}
		
		if(ra > -1 && rc > -1) {
			glVertex3fv((const float *)&a);
			glVertex3fv((const float *)&c);
		}
		
		if(ra > -1 && rd > -1) {
			glVertex3fv((const float *)&a);
			glVertex3fv((const float *)&d);
		}
		
		if(rb > -1 && rc > -1) {
			glVertex3fv((const float *)&b);
			glVertex3fv((const float *)&c);
		}
		
		if(rc > -1 && rd > -1) {
			glVertex3fv((const float *)&c);
			glVertex3fv((const float *)&d);
		}
		
		if(rd > -1 && rb > -1) {
			glVertex3fv((const float *)&d);
			glVertex3fv((const float *)&b);
		}
	}
	glEnd();
}

}
