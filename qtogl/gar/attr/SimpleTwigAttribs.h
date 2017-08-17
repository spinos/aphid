/*
 *  SimpleTwigAttribs.h
 *  
 *  synthesize from a stem and many leaves
 *
 *  Created by jian zhang on 8/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_SIMPLE_TWIG_ATTRIBS_H
#define GAR_SIMPLE_TWIG_ATTRIBS_H

#include "PieceAttrib.h"
#include "MultiSynthesis.h"

namespace aphid {
class Matrix44F;
class SplineMap1D;
}

class SimpleTwigAttribs : public PieceAttrib, public gar::MultiSynthesis {
	
    PieceAttrib* m_inStemAttr;
	PieceAttrib* m_inLeafAttr;
	int m_instId;
	static int sNumInstances;
	
	struct MorphologyParam {
/// angle to local horizontal
		float _petioleAngle;
/// phyllotaxy
		int _leafPlacement;
/// even or odd node
		int _nodeInd;
/// with increment
		float _phyllotaxyAngle;
/// # leaf in ring
		int _whorlCount;
/// to tip		
		float _nodeParam;
		float _deltaNodeParam;
		float _nodeNoiseWeight;
/// sizing to tip
		aphid::SplineMap1D* _sizingSpline;
		aphid::SplineMap1D* _foldingSpline;
		aphid::SplineMap1D* _noiseSpline;
		aphid::SplineMap1D* _agingSpline;
	};
	
	MorphologyParam m_morph;
	
public:
	SimpleTwigAttribs();
	virtual ~SimpleTwigAttribs();
	
	virtual bool update();
/// multi instance of different settings
	virtual int attribInstanceId() const;
/// recv input attr
	virtual void connectTo(PieceAttrib* another, const std::string& portName);
/// when x < 1024 select stem geom x
/// when x >= 1024 select leaf geom x>>10
	virtual aphid::ATriangleMesh* selectGeom(gar::SelectProfile* prof) const;
	
	virtual bool isSynthesized() const;
	virtual int numSynthesizedGroups() const;
	virtual gar::SynthesisGroup* synthesisGroup(int i) const;
	virtual bool canConnectToViaPort(const PieceAttrib* another, const std::string& portName) const;
	
private:
    bool connectToStem(PieceAttrib* another);
	bool connectToLeaf(PieceAttrib* another);
	
	static bool CanBeTwigStem(int x);
	static bool CanBeTwigLeaf(int x);
	
	void synthsizeAGroup(gar::SynthesisGroup* grp,
		const aphid::ATriangleMesh* stemGeom);
	
	void processPhyllotaxy(gar::SynthesisGroup* grp,
			const aphid::Matrix44F& petmat,
			const aphid::Matrix44F& segmat);
	
};

#endif