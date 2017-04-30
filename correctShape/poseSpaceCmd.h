/*
 *  influeOFLCmd.h
 *  opium
 *
 *  Created by jian zhang on 6/14/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include <maya/MGlobal.h>
#include <maya/MPxCommand.h>
#include <maya/MSyntax.h>
#include <maya/MGlobal.h>
#include <maya/MArgDatabase.h>
#include <maya/MVectorArray.h>
#include <maya/MPointArray.h>

class CBPoseSpaceCmd : public MPxCommand
 {
 public:
 				CBPoseSpaceCmd();
	virtual		~CBPoseSpaceCmd();
 	MStatus doIt( const MArgList& args );
 	static void* creator();
	
private:
	MStatus parseArgs ( const MArgList& args );
	void calculateVertexPoseSpace(const MObject& poseMesh, const MObject& baseMesh);
	MString saveResult(const MPointArray& bindPoints, const MPointArray& posePoints, const MVectorArray& dx, const MVectorArray& dy, const MVectorArray& dz);
	MString cacheResult(const MPointArray& bindPoints, const MPointArray& posePoints, const MVectorArray& dx, const MVectorArray& dy, const MVectorArray& dz);
	void setCachedVertexPosition(MObject& poseMesh);
	void saveVertexPosition(MObject& poseMesh);
	void loadVertexPosition(MObject& poseMesh);
	
	MString _cacheName;
	MString _poseName;
	MString _bindName;
	
	enum OperationType {
		tCreate,
		tLoad,
		tSavePose,
		tLoadPose
	};
	
	OperationType _operation;
 };