#pragma once
#include "sargasso_common.h"
#include <string.h>
#include <math.h>
#include <maya/MPxNode.h>
#include <maya/MTypeId.h>
#include <maya/MDataBlock.h>
#include <maya/MDataHandle.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnCompoundAttribute.h>
#include <maya/MTypes.h>

class SargassoNode : public MPxNode
{
public:
						SargassoNode();
	virtual				~SargassoNode(); 

	virtual void		postConstructor();
	virtual MStatus		compute( const MPlug& plug, MDataBlock& data );

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
    
private:
    MPoint m_restPos;
	MVector m_offsetToRest;
	bool m_isInitd;
};
