/*
 *  HesperisAnimIO.h
 *  opium
 *
 *  Created by jian zhang on 10/21/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include "HesperisIO.h"
#include <AAnimationCurve.h>
#include <maya/MFnAnimCurve.h>

namespace aphid {
    
class HesperisAnimIO : public HesperisIO {
public:
	static bool WriteAnimation(const MPlug & attrib, const MObject & animCurveObj);
	
	static bool ReadAnimation(HBase * parent, MObject & entity, MObject & attr);
	static bool ProcessAnimationCurve(const AAnimationCurve & data, MPlug & dst);

	static double SecondsPerFrame;
	static int TangentTypeAsInt(MFnAnimCurve::TangentType type);
	static MFnAnimCurve::TangentType IntAsTangentType(int x);
	
	static AAnimationCurve::CurveType GetAAnimCurveType(MFnAnimCurve::AnimCurveType typ);
	static MFnAnimCurve::AnimCurveType GetMAnimCurveType(AAnimationCurve::CurveType typ);
	static bool RemoveAnimationCurve(MPlug & dst);
};

}