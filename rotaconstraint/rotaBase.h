/*
 *  rotaBase.h
 *  rotaConstraint
 *
 *  Created by jian zhang on 7/7/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <maya/MIOStream.h>
#include <maya/MGlobal.h>
#include <maya/MString.h> 
#include <maya/MPlug.h>
#include <maya/MFnNumericData.h>
#include <maya/MFnTransform.h>
#include <maya/MVector.h>
#include <maya/MDGModifier.h>
#include <maya/MPxTransform.h>
#include <maya/MFnNumericAttribute.h>

class rotaBase {
public:
	enum ConstraintType
	{
		kLargestWeight,
		kSmallestWeight,
	};
};