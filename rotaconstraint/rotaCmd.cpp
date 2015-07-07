/*
 *  rotaCmd.cpp
 *  rotaConstraint
 *
 *  Created by jian zhang on 7/7/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "rotaCmd.h"
#include "geometrySurfaceConstraint.h"
geometrySurfaceConstraintCommand::geometrySurfaceConstraintCommand() {}
geometrySurfaceConstraintCommand::~geometrySurfaceConstraintCommand() {}

void* geometrySurfaceConstraintCommand::creator()
{
	return new geometrySurfaceConstraintCommand();
}

void geometrySurfaceConstraintCommand::createdConstraint(MPxConstraint *constraint)
{
	if ( constraint ) {
		geometrySurfaceConstraint *c = (geometrySurfaceConstraint*) constraint;
		c->weightType = weightType;
	}
	else
		MGlobal::displayError("Failed to get created constraint.");
}

MStatus geometrySurfaceConstraintCommand::parseArgs(const MArgList &argList)
{
	MStatus			ReturnStatus;
	MArgDatabase	argData(syntax(), argList, &ReturnStatus);

	if ( ReturnStatus.error() )
		return MS::kFailure;

	// Settings only work at creation time. Would need an
	// attribute on the node in order to push this state
	// into the node at any time.
	rotaBase::ConstraintType typ;
	if (argData.isFlagSet(kConstrainToLargestWeightFlag))
		typ = rotaBase::kLargestWeight;
	else if (argData.isFlagSet(kConstrainToSmallestWeightFlag))
		typ = rotaBase::kSmallestWeight;
	else
		typ = rotaBase::kLargestWeight;
	weightType = typ;

	// Need parent to process
	return MS::kUnknownParameter;
}

MStatus geometrySurfaceConstraintCommand::doIt(const MArgList &argList)
{
	MStatus ReturnStatus;

	if ( MS::kFailure == parseArgs(argList) )
		return MS::kFailure;
    
	return MS::kUnknownParameter;
}

// MStatus geometrySurfaceConstraintCommand::connectTarget(void *opaqueTarget, int index)
MStatus geometrySurfaceConstraintCommand::connectTarget( MDagPath & targetPath, int index)
{
	MStatus status;/* = connectTargetAttribute(targetPath, index, 
                                            MPxTransform::geometry,
                                            geometrySurfaceConstraint::targetGeometry );
	if (!status) { 
        MGlobal::displayInfo("failed to connectTargetGeometry"); 
        return status;
    }*/
    
    status = connectTargetAttribute(targetPath, index, 
                                    MPxTransform::worldMatrix,
                                    geometrySurfaceConstraint::targetTransform );
	if (!status) {
        MGlobal::displayInfo("failed to connectTargetTransform"); 
        return status;
    }
    
	return MS::kSuccess;
}

MStatus geometrySurfaceConstraintCommand::connectObjectAndConstraint( MDGModifier& modifier )
{
// object to be constrained
	MObject transform = transformObject();
	if ( transform.isNull() ) {
		MGlobal::displayError("Failed to get transformObject()");
		return MS::kFailure;
	}

	MStatus status;
	MFnTransform transformFn( transform );
	MVector translate = transformFn.getTranslation(MSpace::kTransform,&status);
	if (!status) { status.perror(" transformFn.getTranslation"); return status;}

	MPlug translatePlug = transformFn.findPlug( "translate", &status );
	if (!status) { status.perror(" transformFn.findPlug"); return status;}

	if ( MPlug::kFreeToChange == translatePlug.isFreeToChange() ) {
		MFnNumericData nd;
		MObject translateData = nd.create( MFnNumericData::k3Double, &status );
		status = nd.setData3Double( translate.x,translate.y,translate.z);
		if (!status) { status.perror("nd.setData3Double"); return status;}
        
 // set translate value unchanged?
		//status = modifier.newPlugValue( translatePlug, translateData );
		//if (!status) { status.perror("modifier.newPlugValue"); return status;}

		//status = connectObjectAttribute( MPxTransform::geometry, 
		//			geometrySurfaceConstraint::constraintGeometry, false );
		//if (!status) { status.perror("connectObjectAttribute"); return status;}
        
        status = connectObjectAttribute( MPxTransform::translateX, 
		    geometrySurfaceConstraint::constraintTranslateX, false );
		if (!status) { status.perror("connectObjectAttribute tx"); return status;}
        
        status = connectObjectAttribute( MPxTransform::translateY, 
		    geometrySurfaceConstraint::constraintTranslateY, false );
		if (!status) { status.perror("connectObjectAttribute ty"); return status;}
        
        status = connectObjectAttribute( MPxTransform::translateZ, 
		    geometrySurfaceConstraint::constraintTranslateZ, false );
		if (!status) { status.perror("connectObjectAttribute tz"); return status;}
	}

	status = connectObjectAttribute( 
		MPxTransform::parentInverseMatrix,
			geometrySurfaceConstraint::constraintParentInverseMatrix, true, true );
	if (!status) { status.perror("connectObjectAttribute"); return status;}

	return MS::kSuccess;
}

const MObject& geometrySurfaceConstraintCommand::constraintInstancedAttribute() const
{
	return geometrySurfaceConstraint::constraintParentInverseMatrix;
}

const MObject& geometrySurfaceConstraintCommand::constraintOutputAttribute() const
{
	return geometrySurfaceConstraint::constraintGeometry;
}

const MObject& geometrySurfaceConstraintCommand::constraintTargetInstancedAttribute() const
{
	return geometrySurfaceConstraint::targetGeometry;
}

const MObject& geometrySurfaceConstraintCommand::constraintTargetAttribute() const
{
	return geometrySurfaceConstraint::compoundTarget;
}

const MObject& geometrySurfaceConstraintCommand::constraintTargetWeightAttribute() const
{
	return geometrySurfaceConstraint::targetWeight;
}

const MObject& geometrySurfaceConstraintCommand::objectAttribute() const
{ return MPxTransform::geometry; }

MTypeId geometrySurfaceConstraintCommand::constraintTypeId() const
{ return geometrySurfaceConstraint::id; }

// MPxConstraintCommand::TargetType geometrySurfaceConstraintCommand::targetType() const
// { return kGeometryShape; }
// { return kTransform; }

MStatus geometrySurfaceConstraintCommand::appendSyntax()
{
	MStatus ReturnStatus;

	MSyntax theSyntax = syntax(&ReturnStatus);
	if (MS::kSuccess != ReturnStatus) {
		MGlobal::displayError("Could not get the parent's syntax");
		return ReturnStatus;
	}

	// Add our command flags
	theSyntax.addFlag( kConstrainToLargestWeightFlag, kConstrainToLargestWeightFlagLong );
	theSyntax.addFlag( kConstrainToSmallestWeightFlag, kConstrainToSmallestWeightFlagLong );

	return ReturnStatus;
}
