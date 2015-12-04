#include <AHelper.h>
#include "HesMeshNode.h"
#include <maya/MFnMeshData.h>
#include <Vector3F.h>
#include <SHelper.h>

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/progress.hpp"
#include <boost/iostreams/device/file.hpp>
#include <boost/format.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/iostreams/filter/gzip.hpp>

using namespace boost::filesystem;
using namespace std;
namespace io = boost::iostreams;

MTypeId     HesMeshNode::id( 0x29be9231 );
MObject     HesMeshNode::input; 
MObject     HesMeshNode::ameshname;
MObject     HesMeshNode::outMesh;

HesMeshNode::HesMeshNode() {}
HesMeshNode::~HesMeshNode() {}

MStatus HesMeshNode::compute( const MPlug& plug, MDataBlock& data )
{
	
	MStatus stat;
	
	MString cache_name =  data.inputValue( input ).asString();
	std::string substitutedCacheName(cache_name.asChar());
	EnvVar::replace(substitutedCacheName);
	MString mesh_name =  data.inputValue( ameshname ).asString();
	
	if( plug == outMesh ) {
		MDataHandle meshh = data.outputValue(outMesh, &stat);
		
		MFnMeshData dataCreator;
		MObject outMeshData = dataCreator.create(&stat);

		MFnMesh meshFn;
		/*meshFn.create( _numVertex, _numPolygon, vertexArray, polygonCounts, polygonConnects, outMeshData, &stat );
		
		if(_hasUV) {
			stat = meshFn.setUVs ( _uArray, _vArray );
			if(!stat)
			    MGlobal::displayWarning("opium mesh cannot set uvs " + mesh_name);
			stat = meshFn.assignUVs ( polygonCounts, _uvIds );
			if(!stat)
			    MGlobal::displayWarning("opium mesh cannot assugn uvs " + mesh_name);
		}*/
			
		if( !stat ) {
			MGlobal::displayWarning("opium mesh failed to create " + mesh_name);
			return MS::kFailure;
		}


		meshh.set(outMeshData);
	    
		data.setClean(plug);

	} 
	else {
		return MS::kUnknownParameter;
	}

	return MS::kSuccess;
}

void* HesMeshNode::creator()
{
	return new HesMeshNode();
}

MStatus HesMeshNode::initialize()
{
	MFnNumericAttribute numAttr;
	MStatus				stat;
	
	MFnTypedAttribute stringAttr;
	input = stringAttr.create( "hesPath", "hsp", MFnData::kString );
 	stringAttr.setStorable(true);
	addAttribute( input );
	
	ameshname = stringAttr.create( "meshName", "mn", MFnData::kString );
 	stringAttr.setStorable(true);
    stringAttr.setArray(true);
    stringAttr.setDisconnectBehavior(MFnAttribute::kDelete);
	addAttribute( ameshname );
	
	MFnTypedAttribute meshAttr;
	outMesh = meshAttr.create( "outMesh", "o", MFnData::kMesh ); 
	meshAttr.setStorable(false);
	meshAttr.setWritable(false);
    meshAttr.setArray(true);
    meshAttr.setDisconnectBehavior(MFnAttribute::kDelete);
	addAttribute( outMesh );
    
	attributeAffects( input, outMesh );
	
	return MS::kSuccess;
}

MStatus HesMeshNode::connectionMade(const MPlug &plug, const MPlug &otherPlug, bool asSrc)
{
    if ( plug.isElement() ) {
        if( plug.array() == outMesh) {
/// create mesh data 
        }
    }

    return MPxNode::connectionMade( plug, otherPlug, asSrc );
}
//:~
