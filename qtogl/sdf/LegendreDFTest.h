/*
 *  LegendreDFTest.h
 *  sdf
 *
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include <math/Vector3F.h>
#include <math/BoundingBox.h>
#include <math/miscfuncs.h>

namespace aphid {

class GeoDrawer;
class Ray;
struct SuperFormulaParam;
class SuperShapeGlyph;
class ClosestToPointTestResult;

template<typename T, int P, int D>
class LegendreInterpolation;

namespace sdb {

template<typename T>
class VectorArray;

}

namespace cvx {
class Triangle;
}

namespace smp {
class Triangle;
}

template<int I>
class KdNNode;

template<typename T1, typename T2>
class KdNTree;

namespace ttg {
class UniformDensity;
}

}

struct PosSample {
	
	aphid::Vector3F _pos;
	aphid::Vector3F _nml;
	float _r;
	
	aphid::BoundingBox calculateBBox() const {
		return aphid::BoundingBox(_pos.x - _r, _pos.y - _r, _pos.z - _r,
						_pos.x + _r, _pos.y + _r, _pos.z + _r);
	}
	
	static std::string GetTypeStr() {
		return "possamp";
	}
	
	template<typename T>
	void closestToPoint(T * result) const
	{
		aphid::Vector3F tv = _pos - result->_toPoint;
		float d = tv.length();// - _r;
		if(d > aphid::Absolute<float>(result->_distance) ) {
			return;
		}
		
		tv.normalize();
		
		if(_nml.dot(tv) > -.1f ) {
			d = -d;
		}
		
		result->_distance = d;
		result->_hasResult = true;
		result->_hitPoint = _pos;
		result->_hitNormal = _nml;
		
	}
	
};

struct SampleInterp {
	
	bool reject(const PosSample& asmp ) const {
		return false;
	}
	
	void interpolate(PosSample& asmp,
				const float* coord,
				const aphid::cvx::Triangle* g) const {
	
	}
	
};

class LegendreDFTest {

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
	
	aphid::SuperShapeGlyph* m_shape;
	aphid::sdb::VectorArray<aphid::cvx::Triangle>* m_tris;
typedef aphid::KdNTree<PosSample, aphid::KdNNode<4> > TreeTyp;
	TreeTyp * m_tree;
	aphid::ClosestToPointTestResult* m_closestPointTest;
	aphid::sdb::VectorArray<PosSample>* m_pnts;
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
	void drawShape(aphid::GeoDrawer * dr);
	
	void rayIntersect(const aphid::Ray* ray);
	
	aphid::SuperFormulaParam& shapeParam();
	void updateShape();
	void measureShape();
	
private:

	void drawSamples(const float * val, aphid::GeoDrawer * dr) const;
	
	void measureShapeDistance(const aphid::Vector3F& center, const float& scaling);

	void sampleTriangle(PosSample& asmp, aphid::smp::Triangle& sampler, 
							SampleInterp& interp, const aphid::cvx::Triangle* g);
	
	void drawShapeSamples(aphid::GeoDrawer * dr) const;
	void drawAggregatedSamples(aphid::GeoDrawer * dr) const;
	void drawDensity(aphid::GeoDrawer * dr) const;
	void drawFront(aphid::GeoDrawer *dr) const;
	void drawGraph(aphid::GeoDrawer *dr) const;
	void drawError(aphid::GeoDrawer *dr) const;
/// randomly test at sample positions, should close to zero
	void estimateFittingError();
	
};
