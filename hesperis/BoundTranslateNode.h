#pragma once
#include <string.h>
#include <math.h>
#include <maya/MPxNode.h>
#include <maya/MTypeId.h>
#include <maya/MDataBlock.h>
#include <maya/MDataHandle.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnCompoundAttribute.h>
#include <maya/MTypes.h>
#include <maya/MPoint.h>
#include <Vector3F.h>

class BoundTranslateNode : public MPxNode
{
public:
						BoundTranslateNode();
	virtual				~BoundTranslateNode(); 

	virtual void		postConstructor();
	virtual MStatus		compute( const MPlug& plug, MDataBlock& data );

	static  void*		creator();
	static  MStatus		initialize();

public:
	//static MObject		compoundOutput;        
	static MObject		constraintTranslateX;
    static MObject		constraintTranslateY;
    static MObject		constraintTranslateZ;
	
	//static MObject ainBoundMin;
    static MObject ainBoundMinX;
    static MObject ainBoundMinY;
    static MObject ainBoundMinZ;
    //static MObject ainBoundMax;
    static MObject ainBoundMaxX;
    static MObject ainBoundMaxY;
    static MObject ainBoundMaxZ;
	static	MTypeId		id;
private:
    void computeBoundCenter(MDataBlock& block);
private:
	MPoint m_boundCenter;
};
