/*
 *  HesperisAnimIO.cpp
 *  opium
 *
 *  Created by jian zhang on 10/21/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "HesperisAnimIO.h"
#include <AAttributeHelper.h>
#include <HAnimationCurve.h>
#include <AAnimationCurve.h>
#include <maya/MFnAnimCurve.h>
#include <maya/MAnimUtil.h>
#include <boost/format.hpp>

bool HesperisAnimIO::WriteAnimation(const MPlug & attrib, const MObject & animCurveObj, double secondsPerFrame,
								const std::string & beheadName)
{
	MObject entity = attrib.node();
	const std::string nodeName = H5PathNameTo(entity);
	const std::string shortName(attrib.partialName().asChar());
	std::string animName(MFnDependencyNode(animCurveObj).name().asChar());
	SHelper::validateFilePath(animName);
	const std::string attrName = boost::str(boost::format("%1%|%2%|%3%") % nodeName % shortName % animName );
	
		MStatus status;
	MFnAnimCurve animCurve(animCurveObj, &status);
	if (MS::kSuccess != status) {
		AHelper::Info<std::string >(" not an anim curve ", attrName );
		return false;
	}
	
	MFnAnimCurve::AnimCurveType type = animCurve.animCurveType();

	if(type != MFnAnimCurve::kAnimCurveTU ) {
		AHelper::Info<std::string>(" not a TU anim ", attrName );
		return false;
	}
// only weighted for now
	animCurve.setIsWeighted(true);
	
	AHelper::Info<std::string >(" w anim ", attrName );
	
	AAnimationCurve dataCurve;
	dataCurve.setCurveType(AAnimationCurve::TTU);
		
	const unsigned numKeys = animCurve.numKeyframes();
	for (unsigned i = 0; i < numKeys; i++) {
		AAnimationKey dataKey;
		dataKey._key = (float)animCurve.time(i).value() * secondsPerFrame; // as seconds
		// AHelper::Info<float>("time ", dataKey._key);
		dataKey._value = animCurve.value(i);
		
		dataKey._inTangentType = AAttributeHelper::TangentTypeAsShort(animCurve.inTangentType(i));
		dataKey._outTangentType = AAttributeHelper::TangentTypeAsShort(animCurve.outTangentType(i));
		
		MAngle angle;
		double weight;
		animCurve.getTangent(i, angle, weight, true);
		
		dataKey._inAngle = angle.as(MAngle::kDegrees);
		dataKey._inWeight = weight;
		
		animCurve.getTangent(i, angle, weight, false);

		dataKey._outAngle = angle.as(MAngle::kDegrees);
		dataKey._outWeight = weight;
		
		dataCurve.addKey(dataKey);
	}
	
	SaveData<HAnimationCurve, AAnimationCurve>(attrName, &dataCurve);
	
	return true;
}