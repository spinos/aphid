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
	
	const float supsize = supg->gridSize();
	m_mesher.setH(supsize);
	
	const float bsize = supsize * .5f;
	m_pntSz = bsize * .075f;
	BoundingBox cbx;
	
	supg->begin();
	while(!supg->end() ) {
	
		std::vector<Vector3F> smps;
		extractSamplePosIn(smps, supg->value() );
		
		Vector3F center = supg->coordToCellCenter(supg->key() );
		m_mesher.addFrontCell(center, smps);
		
		smps.clear();
		
		supg->getCellBoundingBox(&cbx);
		
		Vector3F bcn = cbx.center();
		float bdx = cbx.distance(0) *.5f;
		float bdy = cbx.distance(1) *.5f;
		float bdz = cbx.distance(2) *.5f;
/// force boundary		
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
	
	m_mesher.buildGrid();

/*
	supg->begin();
	while(!supg->end() ) {
	
		std::vector<Vector3F> smps;
		extractSamplePosIn(smps, supg->value() );
		
		Vector3F center = supg->coordToCellCenter(supg->key() );
		m_mesher.processCell(center, smps);
		
		smps.clear();
		
		supg->next();
	}
*/
	m_mesher.buildMesh();
	m_sampleBegin = m_mesher.numNodes();
	
	m_mesher.setN(m_sampleBegin + sampleGrid()->elementCount() );

	m_mesher.extractGridPosProp();
	extractSamplePos(&m_mesher.X()[m_sampleBegin]);
		
	std::cout<<"\n n node "<<m_sampleBegin
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

#define SHO_SAMPLES 0
#if SHO_SAMPLES
	dr->setColor(0.f, .3f, 0.1f);
	for(i=m_sampleBegin;i<Nv;++i) {
		dr->cube(X[i], m_pntSz);
	}
#endif

#if 1
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
#endif
	
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
	//drawFrontEdges();
	drawFrontTets(dr);
	drawRedBlueGreen(dr);
}

void BccTetrahedralize::drawFrontTets(aphid::GeoDrawer * dr)
{
	Vector3F a, b, c, d;
	int ra, rb, rc, rd;
	const int Nt = m_mesher.numTetrahedrons();
	const Vector3F * X = m_mesher.X();
	const int * prop = m_mesher.prop();
	
	for(int i=0; i<Nt; ++i) {
		const ITetrahedron * t = m_mesher.frontTetrahedron(i, 1, 3);
		if(!t) continue;
		
		a = X[t->iv0];
		b = X[t->iv1];
		c = X[t->iv2];
		d = X[t->iv3];
		
		ra = prop[t->iv0];
		rb = prop[t->iv1];
		rc = prop[t->iv2];
		rd = prop[t->iv3];
		
		dr->tetrahedronWire(a, b, c, d);
	}
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
		const ITetrahedron * t = m_mesher.frontTetrahedron(i, 0, 4);
		if(!t) continue;
		
		a = X[t->iv0];
		b = X[t->iv1];
		c = X[t->iv2];
		d = X[t->iv3];
		
		ra = prop[t->iv0];
		rb = prop[t->iv1];
		rc = prop[t->iv2];
		rd = prop[t->iv3];
		
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

void BccTetrahedralize::drawRedBlueGreen(aphid::GeoDrawer * dr)
{
	const int Nv = m_mesher.N();
	const Vector3F * X = m_mesher.X();
	const int * prop = m_mesher.prop();
	
	dr->m_markerProfile.apply();
	dr->setColor(0.f, 0.f, 0.f);
	int i;
	for(i=0;i<m_sampleBegin;++i) {
		if(prop[i] < 0 )
			continue;
			
		if(prop[i] == 3 ) /// blue
			dr->setColor(0.f, 0.f, 1.f);
		else if(prop[i] == 4 ) /// red
			dr->setColor(1.f, 0.f, 0.f);
		else if(prop[i] == 5 ) /// red-red
			dr->setColor(1.f, .5f, 0.f);
		else if(prop[i] == 6 ) /// blue-blue
			dr->setColor(0.f, .58f, .88f);
		else// if(prop[i] == 7 ) /// red-blue
			dr->setColor(0.79f, 0.f, .89f);	
		
		dr->cube(X[i], m_pntSz);
	}
}

}
