#include "HesUVNode.h"
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnMeshData.h>
#include <maya/MGlobal.h>
#include <maya/MPlug.h>
#include <maya/MDataBlock.h>
#include <maya/MDataHandle.h>
#include <maya/MIOStream.h>
#include <maya/MFnMesh.h>
#include <AHelper.h>
#include <SHelper.h>
#include <HesperisPolygonalMeshIO.h>
#include <baseUtil.h>
#include <APolygonalMesh.h>

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/progress.hpp"
#include <boost/iostreams/device/file.hpp>
#include <boost/format.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/iostreams/filter/gzip.hpp>

using namespace aphid;
using namespace boost::filesystem;
using namespace std;
namespace io = boost::iostreams;

MTypeId     HesUVNode::id( 0xdbd007a );
MObject     HesUVNode::ainput;
MObject     HesUVNode::ameshname;
MObject		 HesUVNode::ainMesh;
MObject		 HesUVNode::aoutMesh;

HesUVNode::HesUVNode() {}

HesUVNode::~HesUVNode() {}

MStatus HesUVNode::compute( const MPlug& plug, MDataBlock& data )
{
	MStatus status = MS::kSuccess;
	MString cacheName =  data.inputValue( ainput ).asString();
	std::string substitutedCacheName(cacheName.asChar());
	EnvVar::replace(substitutedCacheName);
	
	MString meshName = data.inputValue( ameshname ).asString();
	
	if(plug == aoutMesh) {
	
        bool hesStat = false;
        if(BaseUtil::IsImporting)
            hesStat = true;
        else 
            hesStat = BaseUtil::OpenHes(substitutedCacheName, HDocument::oReadOnly);
		
        if(!hesStat) {
			AHelper::Info<std::string >("hes mesh cannot open file ", substitutedCacheName);
			return MS::kFailure;
		}
        
        if(!BaseUtil::HesDoc->find(meshName.asChar())) {
            AHelper::Info<MString>(" hes cannot find mesh ", meshName );
            return MS::kFailure;
		}
        
		HPolygonalMesh entryMesh(meshName.asChar() );
		
		APolygonalMesh dataMesh;
		entryMesh.load(&dataMesh);
        entryMesh.close();
        
        if(!BaseUtil::IsImporting) {
            AHelper::Info<std::string>(" hes mesh close file ", substitutedCacheName );
            BaseUtil::CloseHes();
        }
        
		MDataHandle inputData = data.inputValue( ainMesh, &status );
		MDataHandle outputData = data.outputValue( aoutMesh, &status );
		
        AHelper::Info<MString>( " hes init uv", meshName);
		
		outputData.set(inputData.asMesh());
		
		MObject mesh = outputData.asMesh();

		HesperisMeshUvConnector::appendUV(&dataMesh, mesh);
        
        outputData.setClean();
		
	} else {
		return MS::kUnknownParameter;
	}

	return status;
}

void* HesUVNode::creator()
{
	return new HesUVNode();
}

MStatus HesUVNode::initialize()	
{
	MStatus				status;

	MFnTypedAttribute attrFn;

	ainMesh = attrFn.create("inMesh", "im", MFnMeshData::kMesh);
	attrFn.setStorable(false);
	addAttribute( ainMesh );
    
	aoutMesh = attrFn.create("outMesh", "om", MFnMeshData::kMesh);
	attrFn.setStorable(false);
    attrFn.setWritable(false);
	addAttribute( aoutMesh);
	
	MFnTypedAttribute   stringAttr;
	ainput = stringAttr.create( "hesPath", "hsp", MFnData::kString );
 	stringAttr.setStorable(true);
	addAttribute( ainput );
	
	ameshname = stringAttr.create( "meshName", "mn", MFnData::kString );
 	stringAttr.setStorable(true);
	addAttribute( ameshname );
	
	attributeAffects( ainMesh, aoutMesh );
	
	return MS::kSuccess;
}
