#include <AHelper.h>
#include "HesMeshNode.h"
#include <maya/MFnMeshData.h>
#include <Vector3F.h>
#include <SHelper.h>
#include <BaseUtil.h>
#include <HesperisPolygonalMeshIO.h>

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

HesMeshNode::HesMeshNode() 
{ m_hesStat = false; }

HesMeshNode::~HesMeshNode() {}

MStatus HesMeshNode::compute( const MPlug& plug, MDataBlock& data )
{
	MStatus stat;
	
	MPlug pnames(thisMObject(), ameshname);
	const unsigned numMeshes = pnames.numElements();
	
	MString cacheName =  data.inputValue( input ).asString();
	std::string substitutedCacheName(cacheName.asChar());
	EnvVar::replace(substitutedCacheName);
	
	MArrayDataHandle meshNameArray = data.outputArrayValue( ameshname );
	MArrayDataHandle meshArry = data.outputArrayValue(outMesh, &stat);
	
	if( plug.array() == outMesh ) {
		const unsigned idx = plug.logicalIndex();
		if(!m_hesStat) {
			if(BaseUtil::IsImporting)
				m_hesStat = true;
			else
				m_hesStat = BaseUtil::OpenHes(substitutedCacheName, HDocument::oReadOnly);
		}
		
		if(!m_hesStat) {
			AHelper::Info<std::string >("hes mesh cannot open file ", substitutedCacheName);
			return MS::kFailure;
		}
		
		meshArry.jumpToElement(idx);
		MDataHandle hmesh = meshArry.outputValue();
		
		meshNameArray.jumpToElement(idx);
		const MString meshName = meshNameArray.inputValue().asString();
		
		HPolygonalMesh entryMesh(meshName.asChar() );
		if(!entryMesh.exists()) {
			MGlobal::displayWarning("hes mesh cannot open " + meshName);
			return MS::kFailure;
		}
		
		APolygonalMesh dataMesh;
		entryMesh.load(&dataMesh);
		
		HesperisPolygonalMeshCreator::create(&dataMesh, outMeshData);
		
		MFnMeshData dataCreator;
		MObject outMeshData = dataCreator.create(&stat);
			
		if( !stat ) {
			MGlobal::displayWarning("hes mesh cannot create " + meshName);
			return MS::kFailure;
		}

		hmesh.set(outMeshData);
	    
		data.setClean(plug);
		
		if( (idx+1)>=numMeshes ) {
			if(!BaseUtil::IsImporting) {
				BaseUtil::CloseHes();
				m_hesStat = false;
			}
		}
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
 
        }
    }

    return MPxNode::connectionMade( plug, otherPlug, asSrc );
}
//:~
