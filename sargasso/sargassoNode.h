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
#include <Vector3F.h>
class BaseBuffer;
class ATriangleMesh;
class TriangleDifference;
class SargassoNode : public MPxNode
{
public:
						SargassoNode();
	virtual				~SargassoNode(); 

	virtual void		postConstructor();
	virtual MStatus		compute( const MPlug& plug, MDataBlock& data );

	static  void*		creator();
	static  MStatus		initialize();

    virtual MStatus connectionMade(const MPlug &plug, const MPlug &otherPlug, bool asSrc);
public:
	static MObject		compoundOutput;
    static MObject		targetTransform;
	static MObject		targetWeight;
    static MObject		targetOffset;
    static MObject		targetRestP;
        
	static MObject		aconstraintParentInverseMatrix;

    static MObject		constraintTranslateX;
    static MObject		constraintTranslateY;
    static MObject		constraintTranslateZ;
	
	static MObject atargetRestP;
	static MObject atargetTri;
	static MObject atargetNv;
	static MObject atargetNt;
	static MObject aobjN;
	static MObject aobjLocal;
	static MObject atargetBind;
	static MObject atargetMesh;
	static	MTypeId		id;
private:
    bool creatRestShape(const MObject & m);
    Vector3F * localP();
private:
    ATriangleMesh * m_mesh;
    TriangleDifference * m_diff;
    MPoint m_restPos;
	MVector m_offsetToRest;
    BaseBuffer * m_localP;
	bool m_isInitd;
};
