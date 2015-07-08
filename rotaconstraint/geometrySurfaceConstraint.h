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
    static MObject		targetTransform;
	static MObject		targetGeometry;
	static MObject		targetWeight;
    static MObject		targetOffset;
    static MObject		targetRestP;
        
	static MObject		constraintParentInverseMatrix;
	static MObject		constraintGeometry;
// output translation
    static MObject		constraintTranslateX;
    static MObject		constraintTranslateY;
    static MObject		constraintTranslateZ;
	
	static MObject		constraintTargetX;
    static MObject		constraintTargetY;
    static MObject		constraintTargetZ;
	static MObject		constraintObjectX;
    static MObject		constraintObjectY;
    static MObject		constraintObjectZ;

	static	MTypeId		id;

	rotaBase::ConstraintType weightType;
    
private:
    MPoint m_restPos;
	MVector m_offsetToRest;
	bool m_isInitd;
};

// Useful inline method
inline bool
equivalent(double a, double b  )
{
	return fabs( a - b ) < .000001 ;
}

