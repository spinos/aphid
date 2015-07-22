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
	virtual MStatus connectionBroken(const MPlug &plug, const MPlug &otherPlug, bool asSrc);
public:
	static MObject		compoundOutput;        
	static MObject		aconstraintParentInverseMatrix;
    static MObject		constraintTranslateX;
    static MObject		constraintTranslateY;
    static MObject		constraintTranslateZ;
    static MObject		constraintRotateX;
    static MObject		constraintRotateY;
    static MObject		constraintRotateZ;
	
	static MObject atargetRestP;
	static MObject atargetTri;
	static MObject atargetNv;
	static MObject atargetNt;
	static MObject aobjN;
	static MObject aobjLocal;
    static MObject aobjTri;
	static MObject atargetBind;
	static MObject atargetMesh;
	static	MTypeId		id;
private:
    bool creatRestShape(const MObject & m);
    bool updateShape(const MObject & m);
	void updateSpace(MDataBlock& block, unsigned idx);
    Vector3F * localP();
    unsigned * objectTriangleInd();
private:
	MPoint m_solvedT;
	double m_rot[3];
	MMatrix m_currentSpace;
    ATriangleMesh * m_mesh;
    TriangleDifference * m_diff;
    BaseBuffer * m_localP;
    BaseBuffer * m_triId;
    int m_numObjects;
	bool m_isInitd;
};
