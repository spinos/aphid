/*
 *  LegendreDFTest.h
 *  sdf
 *
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "SvoTest.h"
#include <math/Vector3F.h>
#include <math/BoundingBox.h>
#include <math/miscfuncs.h>
#include <vector>

namespace aphid {

class Ray;
class ClosestToPointTestResult;

template<typename T, int P, int D>
class LegendreInterpolation;

template<int I>
class KdNNode;

template<typename T1, typename T2>
class KdNTree;

}

class LegendreDFTest : public SvoTest {

#define N_L3_DIM 3
#define N_L3_ORD 4
#define N_ORD3 64
#define N_L3_P 3
typedef aphid::LegendreInterpolation<float, 4, 3> PolyInterpTyp;
	PolyInterpTyp* m_poly;

	float m_Yijk[N_ORD3];
	float m_Coeijk[(N_L3_P+1)*(N_L3_P+1)*(N_L3_P+1)];
#define N_SEG 16
#define N_SEG3 4096
	aphid::Vector3F m_samples[N_SEG3];
	float m_errs[N_SEG3];
	float m_exact[N_SEG3];
	float m_appro[N_SEG3];
	
	//aphid::sdb::VectorArray<aphid::cvx::Triangle>* m_tris;
typedef aphid::KdNTree<PosSample, aphid::KdNNode<4> > TreeTyp;
	TreeTyp * m_tree;
	aphid::ClosestToPointTestResult* m_closestPointTest;
	aphid::sdb::FZFCurve* m_fzc;
	aphid::ttg::UniformDensity* m_densityGrid;
	aphid::sdb::VectorArray<PosSample>* m_aggrs;
	
	aphid::Vector3F m_hitP;
	aphid::Vector3F m_hitN;
	aphid::Vector3F m_oriP;
	bool m_isIntersected;

	float m_centerScale[4];
	
public:

	LegendreDFTest();
	virtual ~LegendreDFTest();
	
	virtual bool init();
	virtual void draw(aphid::GeoDrawer * dr);
	
	void rayIntersect(const aphid::Ray* ray);
	
	void measureShape();
	
private:

	void drawSamples(const float * val, aphid::GeoDrawer * dr) const;
	
	void measureShapeDistance(const aphid::Vector3F& center, const float& scaling);

	
	void drawShapeSamples(aphid::GeoDrawer * dr) const;
	void drawAggregatedSamples(aphid::GeoDrawer * dr) const;
	void drawDensity(aphid::GeoDrawer * dr) const;
	void drawFront(aphid::GeoDrawer *dr) const;
	void drawGraph(aphid::GeoDrawer *dr) const;
	void drawError(aphid::GeoDrawer *dr) const;
/// randomly test at sample positions, should close to zero
	void estimateFittingError();
	
};
