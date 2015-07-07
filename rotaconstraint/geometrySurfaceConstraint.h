#pragma once
#include "rotaBase.h"
#include <string.h>
#include <math.h>
#include <maya/MPxConstraint.h>
#include <maya/MTypeId.h>
#include <maya/MDataBlock.h>
#include <maya/MDataHandle.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnCompoundAttribute.h>
#include <maya/MTypes.h>

class geometrySurfaceConstraint : public MPxConstraint
{
public:
						geometrySurfaceConstraint();
	virtual				~geometrySurfaceConstraint(); 

	virtual void		postConstructor();
	virtual MStatus		compute( const MPlug& plug, MDataBlock& data );

	virtual const MObject weightAttribute() const;
    virtual const MObject targetAttribute() const;
	virtual void getOutputAttributes(MObjectArray& attributeArray);

	static  void*		creator();
	static  MStatus		initialize();

public:
	static MObject		compoundTarget;
	static MObject		targetGeometry;
	static MObject		targetWeight;

	static MObject		constraintParentInverseMatrix;
	static MObject		constraintGeometry;

	static	MTypeId		id;

	rotaBase::ConstraintType weightType;
};

// Useful inline method
inline bool
equivalent(double a, double b  )
{
	return fabs( a - b ) < .000001 ;
}

