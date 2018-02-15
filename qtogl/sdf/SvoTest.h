/*
 *  SvoTest.h
 *  
 *
 *  Created by jian zhang on 2/14/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef SVO_TEST_H
#define SVO_TEST_H

#include <math/BoundingBox.h>

namespace aphid {

class GeoDrawer;

struct SuperFormulaParam;
class SuperShapeGlyph;

namespace cvx {
class Triangle;
}

namespace smp {
class Triangle;
}

namespace sdb {

struct FZFCurve;
struct FHilbertRule;

template<typename T>
class SpaceFillingVector;

template<typename T>
class VectorArray;

}

namespace ttg {
class UniformDensity;

template<typename T>
class LegendreSVORule;

struct SVOTNode;

template<typename T>
class SVOTraverser;

template<typename T, typename Tr>
class StackedDrawContext;

template<typename T, typename Tr>
class RaySVOContext;

struct RayHilbertRule;

}

}

struct PosSample;
struct SampleInterp;

class SvoTest {

	aphid::SuperShapeGlyph* m_shape;
	
typedef aphid::ttg::SVOTraverser<aphid::ttg::SVOTNode> OctreeTraverseTyp;
	OctreeTraverseTyp* m_traverser;

typedef aphid::sdb::SpaceFillingVector<PosSample> PntArrTyp;
	PntArrTyp* m_pnts;

typedef aphid::ttg::LegendreSVORule<aphid::sdb::FHilbertRule> SvoRuleTyp;
	SvoRuleTyp* m_hilbertRule;
			
typedef aphid::ttg::StackedDrawContext<aphid::ttg::SVOTNode, SvoRuleTyp > DrawCtxTyp;
	DrawCtxTyp* m_drawCtx;
	
typedef aphid::ttg::RaySVOContext<aphid::ttg::SVOTNode, aphid::ttg::RayHilbertRule > RayCtxTyp;
	RayCtxTyp* m_rayCtx;

public:
	
	SvoTest();
	
	aphid::SuperFormulaParam& shapeParam();
	void updateShape();
	void drawShape(aphid::GeoDrawer * dr);

protected:

	PntArrTyp& pnts();
	const PntArrTyp& pnts() const;
	
	void sampleShape(aphid::BoundingBox& shapeBox);
	
	void drawShapeSamples(aphid::GeoDrawer * dr, const int* drawRange) const;
	
	const aphid::sdb::FHilbertRule& hilbertSFC() const;
	
	SvoRuleTyp& svoRule();
	
	void drawSVO(aphid::GeoDrawer * dr);
	
private:
	
	void sampleTriangle(PosSample& asmp, aphid::smp::Triangle& sampler, 
				SampleInterp& interp, const aphid::cvx::Triangle* g);
	
};

#endif
