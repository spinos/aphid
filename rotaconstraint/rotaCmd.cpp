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
#include <AHelper.h>
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

#ifdef OLD_API
MStatus geometrySurfaceConstraintCommand::connectTarget(void *opaqueTarget, int index)
{
	MGlobal::displayInfo(MString("todo move target to object rotate pivot ")+
		m_objectRotatePvt.x+" "+
		m_objectRotatePvt.y+" "+
		m_objectRotatePvt.z);

	MStatus status = connectTargetAttribute(opaqueTarget, index, 
                                            geometrySurfaceConstraint::targetTransform );
	if (!status) { 
        MGlobal::displayInfo("failed to connectTargetTransform"); 
        return status;
    }
	
	return MS::kSuccess;
}
#else
MStatus geometrySurfaceConstraintCommand::connectTarget( MDagPath & targetPath, int index)
{
	MMatrix ptm = AHelper::GetWorldParentTransformMatrix(targetPath);
	MPoint plocal = m_objectRotatePvt;
	plocal *= ptm.inverse();
    
    MGlobal::displayInfo(MString("move target to object rotate pivot")+
		m_objectRotatePvt.x+" "+
		m_objectRotatePvt.y+" "+
		m_objectRotatePvt.z);
    
    MVector t(plocal.x, plocal.y, plocal.z);
    MFnTransform ftrans(targetPath);
    ftrans.setTranslation(t, MSpace::kTransform);
    
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
    
    MStatus hasAttr;
    MFnNumericAttribute foffset(offsetAttribute(), &hasAttr);
    if(!hasAttr) {
        MGlobal::displayInfo("cannot get constraint target offset attrib");
        return MS::kSuccess;
    }
    foffset.setDefault(m_objectRotatePvt.x, m_objectRotatePvt.y, m_objectRotatePvt.z);
    
	return MS::kSuccess;
}
#endif

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
	
// world rotate pivot
	MDagPath pobj;
	MDagPath::getAPathTo(transform, pobj);
	// MGlobal::displayInfo(MString(" obj p ")+pobj.fullPathName());

	MMatrix wtm = AHelper::GetWorldTransformMatrix(pobj);
	MPoint rotatePivot = transformFn.rotatePivot(MSpace::kTransform);
	
	rotatePivot *= wtm;
	MGlobal::displayInfo(MString("object world rotate pivot ")
                         +rotatePivot.x+" "
                         +rotatePivot.y+" "
                         +rotatePivot.z);

	m_objectRotatePvt = rotatePivot;
	
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
	return geometrySurfaceConstraint::targetTransform;
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

bool geometrySurfaceConstraintCommand::supportsOffset () const
{return true;}

const MObject & geometrySurfaceConstraintCommand::offsetAttribute() const
{ return geometrySurfaceConstraint::targetRestP; }

