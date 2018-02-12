/*
 *  LegendreDFTest.cpp
 *  sdf
 *  
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "LegendreDFTest.h"
#include <math/miscfuncs.h>
#include <math/Calculus.h>
#include <GeoDrawer.h>
#include <ttg/SparseVoxelOctree.h>
#include <ttg/LegendreSVORule.h>
#include "PosSample.h"
#include "measures.h"

LegendreDFTest::LegendreDFTest() 
{ 
	m_fzc = new sdb::FZFCurve;
	m_tree = new KdNTree<PosSample, KdNode4 >();
	m_closestPointTest = new ClosestToPointTestResult;
	m_densityGrid = new ttg::UniformDensity;
	static const float rhoOri[3] = {0.f, 0.f, 0.f};
	m_densityGrid->create(56, 56, 56, rhoOri, 1.f);
	std::cout<<"\n density grid dim "<<m_densityGrid->dimension()[0]
				<<" x "<<m_densityGrid->dimension()[1]
				<<" x "<<m_densityGrid->dimension()[2];
	m_aggrs = new sdb::VectorArray<PosSample>();
	
	m_centerScale[0] = 0;
	m_centerScale[1] = 0;
	m_centerScale[2] = 0;
	m_centerScale[3] = 8.f;
	
	PolyInterpTyp::Initialize();
	
	m_poly = new PolyInterpTyp;
	
}

LegendreDFTest::~LegendreDFTest() 
{}

bool LegendreDFTest::init()
{
	int i,j,k,l;
	int indx[N_L3_DIM];
	
	const float du = 2.f / N_SEG;
	
	for(k=0;k<N_SEG;++k) {
		indx[2] = k;
		for(j=0;j<N_SEG;++j) {
			indx[1] = j;
			for(i=0;i<N_SEG;++i) {
				indx[0] = i;
				
				l = calc::lexIndex(N_L3_DIM, N_SEG, indx);
				
				m_samples[l].set(-1.f + du * (.5f + i),
								-1.f + du * (.5f + j),
								-1.f + du * (.5f + k) );
				
			}
		}
	}

	MeasureSphere d2sphere;
	
	m_poly->compute3 <MeasureSphere> (m_Yijk, m_Coeijk, d2sphere );	
	
	float err, mxErr = 0.f, sumErr = 0.f;
	for(k=0;k<N_SEG;++k) {
		indx[2] = k;
		for(j=0;j<N_SEG;++j) {
			indx[1] = j;
			for(i=0;i<N_SEG;++i) {
				indx[0] = i;
				
				l = calc::lexIndex(N_L3_DIM, N_SEG, indx);
				
				m_exact[l] = d2sphere.measureAt(m_samples[l].x, m_samples[l].y, m_samples[l].z);
				m_appro[l] = m_poly->Approximate3((const float* )&m_samples[l], m_Coeijk );
				
				err = Absolute<float>(m_appro[l] - m_exact[l]);
				m_errs[l] = err;
				if(mxErr < err)
					mxErr = err;
				sumErr += err;
			}
		}
	}
	
	std::cout<<"\n max error "<<mxErr<<" average "<<(sumErr/N_SEG3)
		<<"\n done!";
	std::cout.flush();
	
	measureShape();
	
	return true;
}

void LegendreDFTest::draw(GeoDrawer * dr)
{
	BoundingBox bx(m_centerScale[0] - m_centerScale[3], m_centerScale[1] - m_centerScale[3], m_centerScale[2] - m_centerScale[3],
					m_centerScale[0] + m_centerScale[3], m_centerScale[1] + m_centerScale[3], m_centerScale[2] + m_centerScale[3]);
	dr->boundingBox(bx);

	glPushMatrix();
	
	glTranslatef(m_centerScale[0], m_centerScale[1], m_centerScale[2]);
	glScalef(m_centerScale[3], m_centerScale[3], m_centerScale[3]);
	drawSamples(m_appro, dr);
	
	//drawError(dr);
	
	glPopMatrix();
	
	//drawShapeSamples(dr);
	//drawDensity(dr);
	//drawFront(dr);
	//drawGraph(dr);
	//drawAggregatedSamples(dr);
	drawSVO(dr);
	
	if(m_isIntersected) {
		glColor3f(0.f,1.f,.1f);
		glBegin(GL_LINES);
		glVertex3fv((const float* )&m_oriP);
		glVertex3fv((const float* )&m_hitP);
		glEnd();
		dr->arrow(m_hitP, m_hitP + m_hitN * m_centerScale[3]);
	}
}

void LegendreDFTest::drawSamples(const float * val, GeoDrawer * dr) const
{
	const float ssz = .3f / N_SEG;
	int i=0;
	for(;i<N_SEG3;++i) {
		const float & r = val[i];
		if(Absolute<float>(r) > .1f)
			continue;
/// close to zero
		dr->setColor(0.f,0.f,1.f + r);
		dr->cube(m_samples[i], ssz);
	}
}

void LegendreDFTest::drawError(GeoDrawer *dr) const
{
	const float ssz = .3f / N_SEG;
	int i=0;
	for(;i<N_SEG3;++i) {
		const float & r = m_errs[i];
		if(r < .02f)
			continue;
/// close to zero
		dr->setColor(sqrt(r), 0.f,0.f);
		dr->cube(m_samples[i], ssz);
	}
}

void LegendreDFTest::drawShapeSamples(GeoDrawer * dr) const
{
	const sdb::SpaceFillingVector<PosSample>& rpnts = pnts();
	const int ns = rpnts.size();
	if(ns < 1)
		return;
		
#if 0
	int drawRange[2] = {0,-1};
	rpnts.searchSFC(drawRange, m_fzc->_range);
	SvoTest::drawShapeSamples(dr, drawRange);
#else
	const sdb::FHilbertRule& hil = hilbertSFC();
	SvoTest::drawShapeSamples(dr, hil._range);
#endif
}

void LegendreDFTest::drawAggregatedSamples(GeoDrawer * dr) const
{
	const float& ssz = m_densityGrid->cellSize();
	const int ns = m_aggrs->size();
	glColor3f(.9f,0.f,.7f);
	Vector3F p0;
	glBegin(GL_LINES);
	for(int i=0;i<ns;++i) {
	
		p0 = m_aggrs->get(i)->_pos;
		glVertex3fv((const float* )&p0);
		
		p0 += m_aggrs->get(i)->_nml * ssz;
		glVertex3fv((const float* )&p0);
	}
	glEnd();
}

void LegendreDFTest::drawDensity(GeoDrawer * dr) const
{
	glColor3f(.3f,.1f, .1f);
	const int& m = m_densityGrid->dimension()[0];
	const int& n = m_densityGrid->dimension()[1];
	const int& p = m_densityGrid->dimension()[2];
	for(int k=0;k<p;++k) {
		for(int j=0;j<n;++j) {
			for(int i=0;i<m;++i) {
				if(m_densityGrid->getDensity(i,j,k) > 0) {
					dr->boundingBox(m_densityGrid->getCellBox(i,j,k) );
				}
			}
		}
	}
}

void LegendreDFTest::drawFront(aphid::GeoDrawer *dr) const
{
	glColor3f(0.f,1.f, 0.f);
	BoundingBox frontBox;
	Vector3F frontNml;
	const int& n = m_densityGrid->numFronts();
	for(int i=0;i<n;++i) {
		m_densityGrid->getFront(frontBox, frontNml, i);
		dr->boundingBox(frontBox);
		const Vector3F bc = frontBox.center();
		//dr->arrow(bc, bc + frontNml);
	}
}

void LegendreDFTest::drawGraph(GeoDrawer *dr) const
{
	const float ssz = m_densityGrid->cellSize() * .062f;
	const int& n = m_densityGrid->numNodes();
	for(int i=0;i<n;++i) {
		const DistanceNode& ni = m_densityGrid->nodes()[i];
		if(ni.val < 0.1f)
			dr->setColor(1,0,0);
		else
			dr->setColor(0,ni.val / 10.f,0);
		dr->cube(ni.pos, ssz);
	}
	/*
	const int& ne = m_densityGrid->numEdges();
	glBegin(GL_LINES);
	for(int i=0;i<ne;++i) {
		const IDistanceEdge& ei = m_densityGrid->edges()[i];
		glVertex3fv((const float* )&m_densityGrid->nodes()[ei.vi.x].pos );
		glVertex3fv((const float* )&m_densityGrid->nodes()[ei.vi.y].pos);
	}
	glEnd();*/
}

void LegendreDFTest::rayIntersect(const Ray* ray)
{
	m_isIntersected = false;
	const Vector3F offset(-m_centerScale[0], -m_centerScale[1], -m_centerScale[2]);
	BoundingBox bx(m_centerScale[0] - m_centerScale[3], m_centerScale[1] - m_centerScale[3], m_centerScale[2] - m_centerScale[3],
					m_centerScale[0] + m_centerScale[3], m_centerScale[1] + m_centerScale[3], m_centerScale[2] + m_centerScale[3]);
	const float oneoverscal = 1.f / m_centerScale[3];
	float tmin, tmax;
	if(!bx.intersect(*ray, &tmin, &tmax) )
		return;
	
	TransformAprox tm;
	tm._offset[0] = -m_centerScale[0];
	tm._offset[1] = -m_centerScale[1];
	tm._offset[2] = -m_centerScale[2];
	tm._scaling = 1.f / m_centerScale[3];
	
	m_hitP = ray->travel(tmin);
	m_oriP = ray->m_origin;
	
	tm.setU(m_hitP);
	
	float fd = PolyInterpTyp::Approximate3(tm._u, m_Coeijk);
	if(fd < 0.f)
		return;
		
	int step = 0;
	while(fd > .1e-1f) {
		m_hitP += ray->m_dir * (fd * m_centerScale[3] * .998f);
		if(!bx.isPointInside(m_hitP) )
			return;
			
		tm.setU(m_hitP);
		
		fd = PolyInterpTyp::Approximate3(tm._u, m_Coeijk);
		// std::cout<<"\n d "<<step<<" "<<fd;
		// std::cout.flush();
		step++;
		if(step > 25)
			break;
	}
	
	tm.setU(m_hitP);
		
	PolyInterpTyp::ApproximateGradient3((float* )&m_hitN, fd, tm._u, m_Coeijk);
	m_hitN.normalize();
	m_isIntersected = true;
}

void LegendreDFTest::measureShape()
{
	BoundingBox shapeBox;
	sampleShape(shapeBox);
	
	const Vector3F midP = shapeBox.center();
	const float spanL = shapeBox.getLongestDistance();
	const float spanH = spanL * .5f;
	const Vector3F lowP(midP.x - spanH, 
						midP.y - spanH, 
						midP.z - spanH );
						
	const float rhoSize = shapeBox.getLongestDistance() / 54.f;
	float rhoOri[3];
	rhoOri[0] = lowP.x - rhoSize;
	rhoOri[1] = lowP.y - rhoSize;
	rhoOri[2] = lowP.z - rhoSize;
	m_densityGrid->setOriginAndCellSize(rhoOri, rhoSize);
	
#if 0					
	m_fzc->setOrginSpan(lowP.x, 
						lowP.y, 
						lowP.z,
						spanL);
	pnts().createSFC<sdb::FZFCurve>(*m_fzc);
	
	m_fzc->setRange(512, 512, 256,
					767, 767, 511);
#endif

	const sdb::SpaceFillingVector<PosSample>& rpnts = pnts();
	const int ns = rpnts.size();
	for(int i=0;i<ns;++i) {
		m_densityGrid->accumulate(1.f, rpnts[i]._pos);
	}
	
	m_densityGrid->finish();
	
	std::cout<<"\n n density fronts "<<m_densityGrid->numFronts();
	
	PosSample asmp;
	asmp._r = shapeBox.getLongestDistance() * .00191f;
	
	m_aggrs->clear();
	m_densityGrid->buildSamples<PosSample >(asmp, m_aggrs);
	
	TreeProperty::BuildProfile bf;
	bf._maxLeafPrims = 16;
	
	KdEngine eng;
	eng.buildTree<PosSample, KdNode4, 4>(m_tree, m_aggrs, shapeBox, &bf);
	
	Vector3F cen = shapeBox.center();
	float scaling = shapeBox.getLongestDistance() * .5f;
	
	scaling *= .5f;
	cen.x += scaling;
	cen.y += scaling;
	cen.z += scaling;
	measureShapeDistance(cen, scaling);
	
	m_centerScale[0] = cen.x;
	m_centerScale[1] = cen.y;
	m_centerScale[2] = cen.z;
	m_centerScale[3] = scaling;
	
	estimateFittingError();
	
	std::cout.flush();
}

void LegendreDFTest::measureShapeDistance(const Vector3F& center, const float& scaling)
{
	std::cout<<"\n field bbox center "<<center
		<<" scale "<<scaling;
		
	KdMeasure observer;
	observer._tree = m_tree;
	observer._scaling = scaling;
	observer._offset = center;
	
	m_poly->compute3 <KdMeasure> (m_Yijk, m_Coeijk, observer );	
	
	int i,j,k,l;
	int indx[N_L3_DIM];	
	
	float err, mxErr = 0.f, sumErr = 0.f;
	for(k=0;k<N_SEG;++k) {
		indx[2] = k;
		for(j=0;j<N_SEG;++j) {
			indx[1] = j;
			for(i=0;i<N_SEG;++i) {
				indx[0] = i;
				
				l = calc::lexIndex(N_L3_DIM, N_SEG, indx);
								
				m_exact[l] = observer.measureAt(m_samples[l].x, m_samples[l].y, m_samples[l].z);
				m_appro[l] = PolyInterpTyp::Approximate3((const float* )&m_samples[l], m_Coeijk);
				
				err = Absolute<float>(m_appro[l] - m_exact[l]);
				m_errs[l] = err;
				if(mxErr < err)
					mxErr = err;
				sumErr += err;
			}
		}
	}
	
	std::cout<<"\n max error "<<mxErr<<" average "<<(sumErr/N_SEG3)
		<<"\n done!";
	std::cout.flush();
}

void LegendreDFTest::estimateFittingError()
{
/// to local
	TransformAprox tm;
	tm._offset[0] = -m_centerScale[0];
	tm._offset[1] = -m_centerScale[1];
	tm._offset[2] = -m_centerScale[2];
	tm._scaling = 1.f / m_centerScale[3];
	
	BoundingBox bx(m_centerScale[0] - m_centerScale[3], m_centerScale[1] - m_centerScale[3], m_centerScale[2] - m_centerScale[3],
					m_centerScale[0] + m_centerScale[3], m_centerScale[1] + m_centerScale[3], m_centerScale[2] + m_centerScale[3]);
	
	Vector3F localP;
	const int ns = m_aggrs->size();
	float mxErr = 0.f;
	float sumErr = 0.f;
	int ntest = 125;
	while(ntest > 0) {
	
		const PosSample* si = m_aggrs->get(rand() % ns);
		localP = si->_pos;
		
		if(!bx.isPointInside(localP) )
			continue;

		tm.setU(localP);
		float fd = Absolute<float>(PolyInterpTyp::Approximate3(tm._u, m_Coeijk) );
		if(fd > .13f) {
			std::cout<<"\n max fitting error "<<fd<<" exceeding limit";
			return;
		}
		sumErr += fd;
		if(mxErr < fd)
			mxErr = fd;
			
		ntest--;
	}
	
	std::cout<<"\n max fitting error "<<mxErr<<" average "<<(sumErr / 125.f);
		
}
