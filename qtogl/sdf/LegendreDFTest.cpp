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
#include <math/ANoise3.h>
#include <geom/SuperShape.h>
#include <GeoDrawer.h>
#include <kd/IntersectEngine.h>
#include <kd/ClosestToPointEngine.h>
#include <smp/Triangle.h>
#include <ttg/UniformDensity.h>
#include <math/LegendreInterpolation.h>

using namespace aphid;

struct MeasureSphere {

	float measureAt(const float& x, const float& y, const float& z) {
		
		float cx = x * 1.1f + .1f;
		float cy = y * .9f + .3f;
		float cz = z * .7f + .8f;
		float r = sqrt(cx * cx + cy * cy + cz * cz);
		return r - 1.1f;
	
	}
};

struct MeasureNoise {

	float measureAt(const float & x, const float & y, const float & z) const
	{
		const Vector3F at(x, 1.03f, z);
		const Vector3F orp(-.5421f, -.7534f, -.386f);
		return y - ANoise3::Fbm((const float *)&at,
											(const float *)&orp,
											.7f,
											4,
											1.8f,
											.5f);
	}
};

struct KdMeasure {

	KdEngine _engine;
	ClosestToPointTestResult _ctx;
	KdNTree<PosSample, aphid::KdNNode<4> >* _tree;
	Vector3F _u;
	
	Vector3F _offset;
	float _scaling;
	
	float measureAt(const float& x, const float& y, const float& z) {

/// to world		
		_u.x = x * _scaling;
		_u.y = y * _scaling;
		_u.z = z * _scaling;
		_u += _offset;
	
		_ctx.reset(_u, 1e8f, true);
		_engine.closestToPoint<PosSample>(_tree, &_ctx);
		if(!_ctx._hasResult)
			return 0.f;
/// back to local, which is [-1,1]	
		return _ctx._distance / _scaling;
	}
	
};

struct TransformAprox {

	float _u[3];
	float _scaling;
	float _offset[3];
	
	void setU(const float* p) {
		_u[0] = p[0] + _offset[0];
		_u[1] = p[1] + _offset[1];
		_u[2] = p[2] + _offset[2];
		_u[0] *= _scaling;
		_u[1] *= _scaling;
		_u[2] *= _scaling;
	}
	
	void setU(const Vector3F& p) {
		setU((const float*)&p);
	}
	
};

LegendreDFTest::LegendreDFTest() 
{ 
	m_shape = new SuperShapeGlyph; 
	m_tris = new sdb::VectorArray<cvx::Triangle>();
	m_pnts = new sdb::VectorArray<PosSample>();
	m_tree = new KdNTree<PosSample, KdNode4 >();
	m_closestPointTest = new ClosestToPointTestResult;
	m_densityGrid = new ttg::UniformDensity;
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
	
	drawShapeSamples(dr);
	//drawDensity(dr);
	//drawFront(dr);
	//drawGraph(dr);
	//drawAggregatedSamples(dr);
	
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
	const int ns = m_pnts->size();
	glColor3f(.9f,.6f,0.f);
	glBegin(GL_POINTS);
	for(int i=0;i<ns;++i) {
		glVertex3fv((const float* )&(m_pnts->get(i)->_pos));
	}
	glEnd();
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

void LegendreDFTest::drawShape(GeoDrawer * dr)
{
	dr->triangleMesh(m_shape);
}

SuperFormulaParam& LegendreDFTest::shapeParam()
{ return m_shape->param(); }

void LegendreDFTest::updateShape()
{ m_shape->computePositions(); }

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
	while(fd > .21e-1f) {
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
	m_tris->clear();
	BoundingBox shapeBox;
	shapeBox.reset();
	KdEngine eng;
	eng.appendSource<cvx::Triangle, ATriangleMesh >(m_tris, shapeBox,
									m_shape, 0);
	shapeBox.round();
	
	const float ssz = shapeBox.getLongestDistance() * .0037f;
	smp::Triangle sampler;
	sampler.setSampleSize(ssz);
	
	SampleInterp interp;
	
	m_pnts->clear();
	PosSample asmp;
/// same radius
	asmp._r = ssz;
	const int nt = m_tris->size();
	for(int i=0;i<nt;++i) {
		
		const cvx::Triangle* ti = m_tris->get(i);
		
		sampleTriangle(asmp, sampler, interp, ti);
	}
	
	std::cout<<"\n n triangle samples "<<m_pnts->size();
	
	const float rhoSize = shapeBox.getLongestDistance() / 52.f;
	const int rhoM = shapeBox.distance(0) / rhoSize + 2;
	const int rhoN = shapeBox.distance(1) / rhoSize + 2;
	const int rhoP = shapeBox.distance(2) / rhoSize + 2;
	float rhoOri[3];
	rhoOri[0] = shapeBox.getMin(0) - rhoSize;
	rhoOri[1] = shapeBox.getMin(1) - rhoSize;
	rhoOri[2] = shapeBox.getMin(2) - rhoSize;
	m_densityGrid->create(rhoM, rhoN, rhoP, rhoOri, rhoSize);
	
	std::cout<<"\n density grid dim "<<m_densityGrid->dimension()[0]
				<<" x "<<m_densityGrid->dimension()[1]
				<<" x "<<m_densityGrid->dimension()[2];
	
	const int ns = m_pnts->size();
	for(int i=0;i<ns;++i) {
		m_densityGrid->accumulate(1.f, m_pnts->get(i)->_pos);
	}
	
	m_densityGrid->finish();
	
	std::cout<<"\n n density fronts "<<m_densityGrid->numFronts();
	
	m_aggrs->clear();
	m_densityGrid->buildSamples<PosSample >(asmp, m_aggrs);
	
	TreeProperty::BuildProfile bf;
	bf._maxLeafPrims = 16;
	
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

void LegendreDFTest::sampleTriangle(PosSample& asmp, smp::Triangle& sampler, 
						SampleInterp& interp, const cvx::Triangle* g)
{
	const int ns = sampler.getNumSamples(g->calculateArea() );
	int n = ns;
	for(int i=0;i<500;++i) {
		
		if(n < 1)
			return;
			
		if(!sampler.sampleTriangle<PosSample, SampleInterp >(asmp, interp, g) )
			continue;
			
		m_pnts->insert(asmp);
		n--;
	}
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
