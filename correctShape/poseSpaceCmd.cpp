/*
 *  affectOFLCmd.cpp
 *  opium
 *
 *  Created by jian zhang on 6/14/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "poseSpaceCmd.h"
#include "../shared/SHelper.h"
#include "../opium/aSearchHelper.h"
#include <boost/bind.hpp>
#include <boost/tuple/tuple.hpp> 
#include <algorithm>
class MString;
#include <string>
#include <vector>
using namespace std;
#include <maya/MString.h>
#include <maya/MPlugArray.h>
#include <maya/MArgList.h>

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/file.hpp>

#include "boost/date_time/gregorian/gregorian.hpp"
#include "boost/date_time/posix_time/posix_time.hpp"

using namespace boost::gregorian;
using namespace boost::posix_time;

namespace io = boost::iostreams;
using namespace boost::filesystem;

CBPoseSpaceCmd::CBPoseSpaceCmd() {}

CBPoseSpaceCmd::~CBPoseSpaceCmd() {}

MStatus CBPoseSpaceCmd::doIt( const MArgList& args ) 
{

	MStatus status = parseArgs( args );
	
	if( status != MS::kSuccess ) return status;
	
	ASearchHelper finder;
	
	MObject poseMesh;
	finder.getObjByFullName(_poseName.asChar(), poseMesh);
	
	if(poseMesh == MObject::kNullObj)
	{
		MGlobal::displayWarning(_poseName+" doesn't exist, skipped.");
		return MS::kFailure;
	}
	
	if(!poseMesh.hasFn(MFn::kMesh))
		return MS::kFailure;

	if(_operation == tCreate) {
		MObject bindMesh;
		finder.getObjByFullName(_bindName.asChar(), bindMesh);
			
		if(bindMesh == MObject::kNullObj) {
			MGlobal::displayWarning(_bindName+" doesn't exist, skipped.");
			return MS::kFailure;
		}
		
		if(!bindMesh.hasFn(MFn::kMesh))
			return MS::kFailure;
		
		calculateVertexPoseSpace(poseMesh, bindMesh);	
	}
	else {
		if(!is_regular_file( _cacheName.asChar())) {
			MGlobal::displayInfo(MString("corrective blendshape cannot open pose cache file: ") + _cacheName);
			return MS::kFailure;
		}
		if(_operation == tLoad)
			setCachedVertexPosition(poseMesh);
		else if(_operation == tSavePose)
			saveVertexPosition(poseMesh);
		else if(_operation == tLoadPose)
			loadVertexPosition(poseMesh);
	}
	
return MS::kSuccess;
 }

void* CBPoseSpaceCmd::creator() {
 return new CBPoseSpaceCmd;
 }
 
MStatus CBPoseSpaceCmd::parseArgs( const MArgList& args )
{
	_operation = tCreate;
	_cacheName = "";
	_poseName = "";
	_bindName = "";
	// Parse the arguments.
	MStatus stat = MS::kSuccess;
	MString     	arg;
	const MString	createCacheFlag			("-cc");
	const MString	createCacheFlagLong		("-createCache");
	const MString	loadCacheFlag			("-lc");
	const MString	loadCacheFlagLong		("-loadCache");
	const MString	savePoseFlag			("-sp");
	const MString	savePoseFlagLong		("-savePose");
	const MString	loadPoseFlag			("-lp");
	const MString	loadPoseFlagLong		("-loadPose");
	const MString	poseAtFlag			("-pa");
	const MString	poseAtFlagLong		("-poseAt");
	const MString	bindToFlag			("-bt");
	const MString	bindToFlagLong		("-bindTo");
	for ( unsigned int i = 0; i < args.length(); i++ ) {
		arg = args.asString( i, &stat );
		if (!stat)              
			continue;
				
		if ( arg == createCacheFlag || arg == createCacheFlagLong ) {
			_operation = tCreate;
		}
		else if ( arg == loadCacheFlag || arg == loadCacheFlagLong ) {
			if (i == args.length()-1) 
				continue;
			i++;
			_operation = tLoad;
			args.get(i, _cacheName);
		}
		else if ( arg == savePoseFlag || arg == savePoseFlagLong ) {
			if (i == args.length()-1) 
				continue;
			i++;
			_operation = tSavePose;
			args.get(i, _cacheName);
		}
		else if ( arg == loadPoseFlag || arg == loadPoseFlagLong ) {
			if (i == args.length()-1) 
				continue;
			i++;
			_operation = tLoadPose;
			args.get(i, _cacheName);
		}
		else if ( arg == poseAtFlag || arg == poseAtFlagLong ) {
			if (i == args.length()-1) 
				continue;
			i++;
			args.get(i, _poseName);
		}
		else if ( arg == bindToFlag || arg == bindToFlagLong ) {
			if (i == args.length()-1) 
				continue;
			i++;
			args.get(i, _bindName);
		}
		else {
			MGlobal::displayInfo(MString("unknown flag ") + arg);
		}
	}
	
	if(_operation == tLoad) 
	{
		if( _cacheName == "") {
			MGlobal::displayError("must give -lc cacheFileName to load pose cache");
			return MS::kFailure;
		}
		if(_poseName == "") {
			MGlobal::displayError("must give -pa poseMeshName to create pose cache");
			return MS::kFailure;
		}
	}
	else if(_operation == tSavePose) 
	{
		if(_cacheName == "") {
			MGlobal::displayError("must give -sp cacheFileName to save pose cache");
			return MS::kFailure;
		}
		if(_poseName == "") {
			MGlobal::displayError("must give -pa poseMeshName to save pose cache");
			return MS::kFailure;
		}
	}
	else if(_operation == tLoadPose) 
	{
		if(_cacheName == "") {
			MGlobal::displayError("must give -lp cacheFileName to load pose cache");
			return MS::kFailure;
		}
		if(_poseName == "") {
			MGlobal::displayError("must give -pa poseMeshName to load pose cache");
			return MS::kFailure;
		}
	}
	else if(_operation == tCreate) {
		if(_poseName == "" || _bindName == "") {
			MGlobal::displayError("must give -pa poseMeshName and -bt bindMeshName to create pose cache");
			return MS::kFailure;
		}
	}
	
	return stat;
}

void CBPoseSpaceCmd::calculateVertexPoseSpace(const MObject& poseMesh, const MObject& bindMesh)
{
	MStatus status;
	MFnMesh poseFn(poseMesh, &status);
	MPointArray originalPoseVertex;
	poseFn.getPoints ( originalPoseVertex, MSpace::kObject );
	
	unsigned numVert = originalPoseVertex.length();
	
	MFnMesh bindFn(bindMesh, &status);
	MPointArray originalbindVertex;
	bindFn.getPoints ( originalbindVertex, MSpace::kObject );
	
	MPointArray movedVertex(originalbindVertex);
	for(unsigned i=0; i < numVert; i++)
	{
		movedVertex[i].x =  originalbindVertex[i].x + 1.0;
	}
	
	bindFn.setPoints(movedVertex, MSpace::kObject );
	bindFn.updateSurface();
	poseFn.updateSurface();
	
	MPointArray xPoseVertex;
	poseFn.getPoints ( xPoseVertex, MSpace::kObject );
	
	MVectorArray dirx;
	dirx.setLength(numVert);
	
	for(unsigned i=0; i < numVert; i++) {
		dirx[i] = xPoseVertex[i] - originalPoseVertex[i];
	}
	
	for(unsigned i=0; i < numVert; i++) {
		movedVertex[i].x =  originalbindVertex[i].x;
		movedVertex[i].y =  originalbindVertex[i].y + 1.0;
	}
	
	bindFn.setPoints(movedVertex, MSpace::kObject );
	bindFn.updateSurface();
	poseFn.updateSurface();
	
	poseFn.getPoints ( xPoseVertex, MSpace::kObject );
	
	MVectorArray diry;
	diry.setLength(numVert);
	
	for(unsigned i=0; i < numVert; i++) {
		diry[i] = xPoseVertex[i] - originalPoseVertex[i];
	}

	for(unsigned i=0; i < numVert; i++) {
		movedVertex[i].y =  originalbindVertex[i].y;
		movedVertex[i].z =  originalbindVertex[i].z + 1.0;
	}
	
	bindFn.setPoints(movedVertex, MSpace::kObject );
	bindFn.updateSurface();
	poseFn.updateSurface();
	
	poseFn.getPoints ( xPoseVertex, MSpace::kObject );
	
	MVectorArray dirz;
	dirz.setLength(numVert);
	
	for(unsigned i=0; i < numVert; i++) {
		dirz[i] = xPoseVertex[i] - originalPoseVertex[i];
	}
	
	bindFn.setPoints(originalbindVertex, MSpace::kObject );
	bindFn.updateSurface();
	
	const MString cacheNodeName = cacheResult(originalbindVertex, originalPoseVertex, dirx, diry, dirz);
	
	MGlobal::displayInfo(MString("correct shape recordes ") + numVert + " points in " + cacheNodeName);
    	
	setResult(cacheNodeName);
}

MString CBPoseSpaceCmd::saveResult(const MPointArray& bindPoints, const MPointArray& posePoints, const MVectorArray& dx, const MVectorArray& dy, const MVectorArray& dz)
{
	unsigned count = dx.length();
	float* data = new float[count * (3 + 3 + 16)];
	
	for(unsigned i=0; i < count; i++)
	{
		data[i*3] = bindPoints[i].x;
		data[i*3+1] = bindPoints[i].y;
		data[i*3+2] = bindPoints[i].z;
	}
	
	unsigned offset = count * 3;
	
	for(unsigned i=0; i < count; i++)
	{
		data[offset+i*3] = posePoints[i].x;
		data[offset+i*3+1] = posePoints[i].y;
		data[offset+i*3+2] = posePoints[i].z;
	}
	
	offset = count * 6;
	
	for(unsigned i=0; i < count; i++)
	{
		float m[4][4];
		
		m[0][0] = dx[i].x;
		m[0][1] = dx[i].y;
		m[0][2] = dx[i].z;
		m[0][3] = 0.f;
		m[1][0] = dy[i].x;
		m[1][1] = dy[i].y;
		m[1][2] = dy[i].z;
		m[1][3] = 0.f;
		m[2][0] = dz[i].x;
		m[2][1] = dz[i].y;
		m[2][2] = dz[i].z;
		m[2][3] = 0.f;
		m[3][0] = 0.f;
		m[3][1] = 0.f;
		m[3][2] = 0.f;
		m[3][3] = 1.f;
		
		MMatrix tm(m);
		tm = tm.inverse();
		
		tm.get(m);
		
		unsigned ivx = offset+i*16;
		
		data[ivx]    = m[0][0];
		data[ivx+1]  = m[0][1];
		data[ivx+2]  = m[0][2];
		data[ivx+3]  = m[0][3];
		data[ivx+4]  = m[1][0];
		data[ivx+5]  = m[1][1];
		data[ivx+6]  = m[1][2];
		data[ivx+7]  = m[1][3];
		data[ivx+8]  = m[2][0];
		data[ivx+9]  = m[2][1];
		data[ivx+10] = m[2][2];
		data[ivx+11] = m[2][3];
		data[ivx+12] = m[3][0];
		data[ivx+13] = m[3][1];
		data[ivx+14] = m[3][2];
		data[ivx+15] = m[3][3];
	}
	
	io::filtering_ostream out;
	out.push(boost::iostreams::gzip_compressor());
	
	MString filename = _cacheName;
	
	if(filename == "") {
	
	MString projRoot;
	MGlobal::executeCommand(MString("workspace -q -dir"), projRoot, 0, 0);
	
	projRoot = projRoot + "/poses/";
	if(!exists( projRoot.asChar() )) {
		create_directory(projRoot.asChar());
	}
	
	const ptime now = second_clock::local_time();
	std::string file_time = to_iso_string(now);
	
		filename = projRoot+file_time.c_str()+".pos";
	}
	
	out.push(io::file_sink(filename.asChar(), ios::binary));

	out.write((char*)data, count*(3 + 3 + 16)*4);
	out.flush();
	
	delete[] data;
	
	MGlobal::displayInfo(MString("corrective blendshape writes pose cache to ") + filename);
	return filename;
}

MString CBPoseSpaceCmd::cacheResult(const MPointArray& bindPoints, const MPointArray& posePoints, const MVectorArray& dx, const MVectorArray& dy, const MVectorArray& dz)
{
	MDGModifier modif;
	MObject opose = modif.createNode("sculptSpaceRecord");
	modif.doIt();

    unsigned count = dx.length();

    MVectorArray row0Array;
    row0Array.setLength(count);
    MVectorArray row1Array;
    row1Array.setLength(count);
    MVectorArray row2Array;
    row2Array.setLength(count);
    MVectorArray row3Array;
    row3Array.setLength(count);
    
    MVectorArray bndArray;
    bndArray.setLength(count);
    
    MVectorArray posArray;
    posArray.setLength(count);
	
	float m[4][4];
    
    for(unsigned i=0; i < count; i++) {
		m[0][0] = dx[i].x;
		m[0][1] = dx[i].y;
		m[0][2] = dx[i].z;
		m[0][3] = 0.f;
		m[1][0] = dy[i].x;
		m[1][1] = dy[i].y;
		m[1][2] = dy[i].z;
		m[1][3] = 0.f;
		m[2][0] = dz[i].x;
		m[2][1] = dz[i].y;
		m[2][2] = dz[i].z;
		m[2][3] = 0.f;
		m[3][0] = 0.f;
		m[3][1] = 0.f;
		m[3][2] = 0.f;
		m[3][3] = 1.f;
		
		MMatrix tm(m);
		tm = tm.inverse();
		
		tm.get(m);
		
		row0Array[i].x = m[0][0];
		row0Array[i].y = m[0][1];
		row0Array[i].z = m[0][2];
		row1Array[i].x = m[1][0];
		row1Array[i].y = m[1][1];
		row1Array[i].z = m[1][2];
		row2Array[i].x = m[2][0];
		row2Array[i].y = m[2][1];
		row2Array[i].z = m[2][2];
		row3Array[i].x = m[3][0];
		row3Array[i].y = m[3][1];
		row3Array[i].z = m[3][2];
		
		bndArray[i] = bindPoints[i];
		posArray[i] = posePoints[i];
	}
	
	MFnDependencyNode fposec(opose);
	
	MStatus stat;
	MPlug pspacerow0 = fposec.findPlug("poseSpaceRow0", false, &stat);
	MPlug pspacerow1 = fposec.findPlug("poseSpaceRow1", false, &stat);
	MPlug pspacerow2 = fposec.findPlug("poseSpaceRow2", false, &stat);
	MPlug pspacerow3 = fposec.findPlug("poseSpaceRow3", false, &stat);
	MPlug pbind = fposec.findPlug("bpnt", false, &stat);
	MPlug ppose = fposec.findPlug("ppnt", false, &stat);
	
	MFnVectorArrayData frow0;
    MObject orow0 = frow0.create(row0Array);
    pspacerow0.setMObject(orow0);
    
    MFnVectorArrayData frow1;
    MObject orow1 = frow1.create(row1Array);
    pspacerow1.setMObject(orow1);
    
    MFnVectorArrayData frow2;
    MObject orow2 = frow2.create(row2Array);
    pspacerow2.setMObject(orow2);
    
    MFnVectorArrayData frow3;
    MObject orow3 = frow3.create(row3Array);
    pspacerow3.setMObject(orow3);
    
    MFnVectorArrayData fbind;
    MObject obind = fbind.create(bndArray);
    pbind.setMObject(obind);
    
    MFnVectorArrayData fpose;
    MObject oposed = fpose.create(posArray);
    ppose.setMObject(oposed);
    
	return fposec.name();
}

void CBPoseSpaceCmd::setCachedVertexPosition(MObject& poseMesh)
{
	MFnMesh fpose(poseMesh);
	unsigned numVertex = fpose.numVertices();
	
	float* data = new float[numVertex * (3 + 3 + 16)];
	
	boost::iostreams::filtering_istream in;
	in.push( boost::iostreams::gzip_decompressor());
	in.push( boost::iostreams::file_source(_cacheName.asChar(), ios::binary));
	if(!in.read((char*)data, numVertex * (3 + 3 + 16) * 4)) {
		MGlobal::displayError(MString("corrective blendshape failed to read enough data from pose cache ") + _cacheName);
		delete[] data;
		return;
	}
	
	MPointArray sculptVertex;
	
	sculptVertex.setLength(numVertex);

	unsigned offset = numVertex * 3;
	
	for(unsigned i=0; i < numVertex; i++)
	{
		sculptVertex[i].x = data[offset + i*3];
		sculptVertex[i].y = data[offset + i*3+1];
		sculptVertex[i].z = data[offset + i*3+2];
	}
	
	fpose.setPoints(sculptVertex, MSpace::kObject );
	fpose.updateSurface();
	
	delete[] data;
	MGlobal::displayInfo(MString("corrective blendshape read pose cache from ") + _cacheName);
		
}

void CBPoseSpaceCmd::saveVertexPosition(MObject& poseMesh)
{
	MStatus status;
	MFnMesh poseFn(poseMesh, &status);
	MPointArray poseVertex;
	poseFn.getPoints ( poseVertex, MSpace::kObject );
	
	unsigned numVert = poseVertex.length();
	
	float* data = new float[numVert * 3];
	
	for(unsigned i=0; i < numVert; i++)
	{
		data[i*3] = poseVertex[i].x;
		data[i*3+1] = poseVertex[i].y;
		data[i*3+2] = poseVertex[i].z;
	}
	
	io::filtering_ostream out;
	out.push(boost::iostreams::gzip_compressor());
	
	MString filename = _cacheName + ".pose";
	
	out.push(io::file_sink(filename.asChar(), ios::binary));

	out.write((char*)data, numVert *3 * 4);
	out.flush();
	
	delete[] data;
	
	MGlobal::displayInfo(MString("corrective blendshape write sculpt pose to ") + filename);
}

void CBPoseSpaceCmd::loadVertexPosition(MObject& poseMesh)
{
	MStatus status;
	MFnMesh poseFn(poseMesh, &status);
	unsigned numVertex = poseFn.numVertices();
	
	float* data = new float[numVertex * 3];
	
	MString filename = _cacheName + ".pose";
	
	if(!is_regular_file(filename.asChar()))
		return;
	
	boost::iostreams::filtering_istream in;
	in.push( boost::iostreams::gzip_decompressor());
	in.push( boost::iostreams::file_source(filename.asChar(), ios::binary));
	if(!in.read((char*)data, numVertex * 3  * 4)) {
		MGlobal::displayError(MString("corrective blendshape failed to read enough data from pose cache ") + _cacheName);
		delete[] data;
		return;
	}

	
	MPointArray poseVertex;
	poseVertex.setLength(numVertex);
	
	for(unsigned i=0; i < numVertex; i++)
	{
		poseVertex[i].x = data[i*3];
		poseVertex[i].y = data[i*3+1];
		poseVertex[i].z = data[i*3+2];
	}
	
	poseFn.setPoints ( poseVertex, MSpace::kObject );
	delete[] data;
	
	MGlobal::displayInfo(MString("corrective blendshape read sculpt pose from ") + filename);
}
//:~
