/*
 *  rotaCmd.h
 *  rotaConstraint
 *
 *  Created by jian zhang on 7/7/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include "rotaBase.h"
#include <maya/MPxConstraintCommand.h> 
#include <maya/MArgDatabase.h>

#define kConstrainToLargestWeightFlag "-lw"
#define kConstrainToLargestWeightFlagLong "-largestWeight"
#define kConstrainToSmallestWeightFlag "-sw"
#define kConstrainToSmallestWeightFlagLong "-smallestWeight"

class geometrySurfaceConstraintCommand : public MPxConstraintCommand 
{
public:

	geometrySurfaceConstraintCommand();
	~geometrySurfaceConstraintCommand();

	virtual MStatus		doIt(const MArgList &argList);
	virtual MStatus		appendSyntax();

	virtual MTypeId constraintTypeId() const;
    virtual bool 	supportsOffset () const;
    virtual const MObject & 	offsetAttribute () const;
	virtual const MObject& constraintInstancedAttribute() const;
	virtual const MObject& constraintOutputAttribute() const;
	virtual const MObject& constraintTargetInstancedAttribute() const;
	virtual const MObject& constraintTargetAttribute() const;
	virtual const MObject& constraintTargetWeightAttribute() const;
	virtual const MObject& objectAttribute() const;

#ifdef OLD_API
	virtual MStatus connectTarget(void *opaqueTarget, int index);
#else
    virtual MStatus connectTarget(	MDagPath & 	targetPath, int index);
#endif
	virtual MStatus connectObjectAndConstraint( MDGModifier& modifier );

	virtual void createdConstraint(MPxConstraint *constraint);

	static  void* creator();

protected:
	virtual MStatus			parseArgs(const MArgList &argList);
	MPoint m_objectRotatePvt;
	rotaBase::ConstraintType weightType;
};