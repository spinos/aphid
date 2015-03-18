#pragma once

#include <maya/MPxNode.h> 
#include <maya/MFnNumericAttribute.h>
#include <maya/MTypeId.h> 
#include <maya/MIntArray.h>
#include <maya/MVectorArray.h>
#include <maya/MPoint.h>
#include <maya/MFnDagNode.h>
#include <iostream>
#include <fstream>
#include <string>
#include <maya/MFnCamera.h>
#include <maya/MDagPath.h>
#include "ExrImgData.h"
#include <CUDABuffer.h>
#include <map>
class StripeCompressedRGBAZImage;
class ExrImgLoader : public MPxNode
{
public:
						  ExrImgLoader();
	virtual				  ~ExrImgLoader(); 

	virtual MStatus		  compute( const MPlug& plug, MDataBlock& data );
	static  void*		creator();
	static  MStatus		initialize();

public:

	static  	MTypeId		id;
	
	static  	MObject 	adepthImageName;
	static MObject aframeNumber;
    static MObject aframePadding;
    static  	MObject 	aenableMultiFrames;
    
	static MObject aoutval;
	
private:
	void preLoadImage(ExrImgData::DataDesc * dst, const char * name, int frame, int padding, bool useImageSequence);
private:
    static std::map<std::string, StripeCompressedRGBAZImage *> CachedCompressedFiles;
    ExrImgData::DataDesc *_pDesc;
};
