#include "../shared/AHelper.h"
#include "correctBlendShapeNode.h"
#include <maya/MAnimControl.h>
#include <maya/MFnNurbsSurface.h>
#include <maya/MItSurfaceCV.h>
#include <maya/MItGeometry.h>
#include <maya/MItMeshPolygon.h>
#include <maya/MItMeshVertex.h>
#include <maya/MItMeshEdge.h>
#include <maya/MFnMesh.h>
#include <maya/MFnMeshData.h>

#include "../shared/SHelper.h"

#include <boost/format.hpp>

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/progress.hpp"

#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/file.hpp>

#include <iostream>
#include <fstream>

namespace io = boost::iostreams;

using namespace boost::filesystem;

MTypeId     CorrectBlendShapeNode::id( 0x00d29638 );
MObject     CorrectBlendShapeNode::asculptMesh; 
MObject     CorrectBlendShapeNode::aposefile;
MObject CorrectBlendShapeNode::aspacerow0;
MObject CorrectBlendShapeNode::aspacerow1;
MObject CorrectBlendShapeNode::aspacerow2;
MObject CorrectBlendShapeNode::aspacerow3;
MObject CorrectBlendShapeNode::abindpnt;
MObject CorrectBlendShapeNode::aposepnt;
MObject     CorrectBlendShapeNode::outMesh; 

CorrectBlendShapeNode::CorrectBlendShapeNode():_isCached(0)
{
}

CorrectBlendShapeNode::~CorrectBlendShapeNode() 
{
}

char CorrectBlendShapeNode::readInternalCache(MDataBlock& block, unsigned numVertex)
{
    MStatus status;
    MDataHandle hrow0 = block.inputValue(aspacerow0, &status);
    MFnVectorArrayData frow0(hrow0.data(), &status);
    MVectorArray row0Array = frow0.array();

    MDataHandle hrow1 = block.inputValue(aspacerow1, &status);
    MFnVectorArrayData frow1(hrow1.data(), &status);
    MVectorArray row1Array = frow1.array();	
    
    MDataHandle hrow2 = block.inputValue(aspacerow2, &status);
    MFnVectorArrayData frow2(hrow2.data(), &status);
    MVectorArray row2Array = frow2.array();
    
    MDataHandle hbnd = block.inputValue(abindpnt, &status);
    MFnVectorArrayData fbnd(hbnd.data(), &status);
    MVectorArray bndArray = fbnd.array();
    
    MDataHandle hpos = block.inputValue(aposepnt, &status);
    MFnVectorArrayData fpos(hpos.data(), &status);
    MVectorArray posArray = fpos.array();
    
    unsigned numRow0 = row0Array.length();
    unsigned numRow1 = row1Array.length();
    unsigned numRow2 = row2Array.length();
    unsigned numBind = bndArray.length();
    unsigned numPose = posArray.length();
    
    if(numRow0 != numVertex) {
        MGlobal::displayInfo(MString("row0 count ") + numRow0);
        return 0;
    }
    
    if(numRow1 != numVertex) {
        MGlobal::displayInfo(MString("row1 count ") + numRow1);
        return 0;
    }
    
    if(numRow2 != numVertex) {
        MGlobal::displayInfo(MString("row2 count ") + numRow2);
        return 0;
    }
    
    if(numBind != numVertex) {
        MGlobal::displayInfo(MString("bind count ") + numBind);
        return 0;
    }
    
    if(numPose != numVertex) {
        MGlobal::displayInfo(MString("pose count ") + numPose);
        return 0;
    }
    
    _bindPoseVertex.setLength(numVertex);
	_sculptPoseVertex.setLength(numVertex);
	_poseSpace.setLength(numVertex);
	
	float m[4][4];
	for(unsigned i=0; i < numVertex; i++) {
		_bindPoseVertex[i] = bndArray[i];
		_sculptPoseVertex[i] = posArray[i];
		
		m[0][0] = row0Array[i].x;
		m[0][1] = row0Array[i].y;
		m[0][2] = row0Array[i].z;
		m[0][3] = 0.f;
		m[1][0] = row1Array[i].x;
		m[1][1] = row1Array[i].y;
		m[1][2] = row1Array[i].z;
		m[1][3] = 0.f;
		m[2][0] = row2Array[i].x;
		m[2][1] = row2Array[i].y;
		m[2][2] = row2Array[i].z;
		m[2][3] = 0.f;
		m[3][0] = 0.f;
		m[3][1] = 0.f;
		m[3][2] = 0.f;
		m[3][3] = 1.f;
		
		_poseSpace[i] = MMatrix(m);
	}
    
    return 1;
}

char CorrectBlendShapeNode::loadCache(const char* filename, unsigned numVertex)
{
	if(!is_regular_file( filename ))
		return 0;
		
	float* data = new float[numVertex * (3 + 3 + 16)];
	
	boost::iostreams::filtering_istream in;
	in.push( boost::iostreams::gzip_decompressor());
	in.push( boost::iostreams::file_source(filename, ios::binary));
	in.read((char*)data, numVertex * (3 + 3 + 16) * 4);
	
	
	_bindPoseVertex.setLength(numVertex);
	_sculptPoseVertex.setLength(numVertex);
	_poseSpace.setLength(numVertex);
	
	for(unsigned i=0; i < numVertex; i++)
	{
		_bindPoseVertex[i].x = data[i*3];
		_bindPoseVertex[i].y = data[i*3+1];
		_bindPoseVertex[i].z = data[i*3+2];
	}
	
	unsigned offset = numVertex * 3;
	
	for(unsigned i=0; i < numVertex; i++)
	{
		_sculptPoseVertex[i].x = data[offset + i*3];
		_sculptPoseVertex[i].y = data[offset + i*3+1];
		_sculptPoseVertex[i].z = data[offset + i*3+2];
	}
	
	offset = numVertex * 6;
	
	for(unsigned i=0; i < numVertex; i++)
	{
		float m[4][4];
		unsigned ivx = offset+i*16;
		
		m[0][0] = data[ivx];
		m[0][1] = data[ivx+1];
		m[0][2] = data[ivx+2];
		m[0][3] = data[ivx+3];
		m[1][0] = data[ivx+4];
		m[1][1] = data[ivx+5];
		m[1][2] = data[ivx+6];
		m[1][3] = data[ivx+7];
		m[2][0] = data[ivx+8];
		m[2][1] = data[ivx+9];
		m[2][2] = data[ivx+10];
		m[2][3] = data[ivx+11];
		m[3][0] = data[ivx+12];
		m[3][1] = data[ivx+13];
		m[3][2] = data[ivx+14];
		m[3][3] = data[ivx+15];
		
		_poseSpace[i] = MMatrix(m);
	
	}
	
	delete[] data;
			
	return 1;
}

MStatus CorrectBlendShapeNode::compute( const MPlug& plug, MDataBlock& data )
{
	MStatus stat;
	
	if (plug == outMesh)
	{
		MObject sculpted = data.inputValue( asculptMesh, &stat ).asMesh();
		MFnMesh fsculpt(sculpted);
		unsigned numVertex = fsculpt.numVertices();
		MPointArray sculptVertex;
		fsculpt.getPoints(sculptVertex, MSpace::kObject);
		MString filename =  data.inputValue( aposefile ).asString();
		
		if(!_isCached) {
		    _isCached = readInternalCache(data, numVertex);
		}
		
		if(!_isCached && filename != "")
		{ 
			_isCached = loadCache(filename.asChar(), numVertex);
		}

		if(_isCached)
		{
			MPointArray targetVertex;
			targetVertex.setLength(numVertex);
			for(unsigned i = 0; i < numVertex; i++)
			{
				targetVertex[i] = _bindPoseVertex[i];

				double dx = sculptVertex[i].x - _sculptPoseVertex[i].x;
				double dy = sculptVertex[i].y - _sculptPoseVertex[i].y;
				double dz = sculptVertex[i].z - _sculptPoseVertex[i].z;
				
				MPoint disp(dx, dy, dz);
				
				disp *= _poseSpace[i];
				
				targetVertex[i].x += disp.x;
				targetVertex[i].y += disp.y;
				targetVertex[i].z += disp.z;

			}
			fsculpt.setPoints(targetVertex);
		}

		MDataHandle outputData = data.outputValue( outMesh, &stat );
		outputData.set(sculpted);
		data.setClean(plug);
	}

	return MS::kSuccess;
}

void* CorrectBlendShapeNode::creator()
{
	return new CorrectBlendShapeNode();
}

MStatus CorrectBlendShapeNode::initialize()
{
	MFnTypedAttribute tAttr;
	MStatus				stat;
	
	asculptMesh = tAttr.create( "sculptMesh", "sm", MFnData::kMesh );
	addAttribute(asculptMesh);

	aposefile = tAttr.create( "poseFile", "pf", MFnData::kString );
 	tAttr.setStorable(true);
	addAttribute( aposefile );
	
	aspacerow0 = tAttr.create( "poseSpaceRow0", "psr0", MFnData::kVectorArray);
 	tAttr.setStorable(false);
 	tAttr.setWritable(true);
 	tAttr.setReadable(true);
	addAttribute(aspacerow0);
	
	aspacerow1 = tAttr.create( "poseSpaceRow1", "psr1", MFnData::kVectorArray);
 	tAttr.setStorable(false);
 	tAttr.setWritable(true);
 	tAttr.setReadable(true);
	addAttribute(aspacerow1);
	
	aspacerow2 = tAttr.create( "poseSpaceRow2", "psr2", MFnData::kVectorArray);
 	tAttr.setStorable(false);
 	tAttr.setWritable(true);
 	tAttr.setReadable(true);
	addAttribute(aspacerow2);
	
	aspacerow3 = tAttr.create( "poseSpaceRow3", "psr3", MFnData::kVectorArray);
 	tAttr.setStorable(false);
 	tAttr.setWritable(true);
 	tAttr.setReadable(true);
	addAttribute(aspacerow3);
	
	abindpnt = tAttr.create( "bindPoint", "bpnt", MFnData::kVectorArray);
 	tAttr.setStorable(false);
 	tAttr.setWritable(true);
 	tAttr.setReadable(true);
	addAttribute(abindpnt);
	
	aposepnt = tAttr.create( "posePoint", "ppnt", MFnData::kVectorArray);
 	tAttr.setStorable(false);
 	tAttr.setWritable(true);
 	tAttr.setReadable(true);
	addAttribute(aposepnt);
	
	outMesh = tAttr.create("outMesh", "om", MFnData::kMesh);
	tAttr.setStorable(false);
	tAttr.setConnectable(true);
	addAttribute( outMesh );
    
	attributeAffects( asculptMesh, outMesh );

	return MS::kSuccess;
}
//:~
