#include "AdaptiveFieldDeformer.h"
// Function Sets
//
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnSingleIndexedComponent.h>
#include <maya/MString.h>
#include <maya/MFloatArray.h>
#include <maya/MPointArray.h>
#include <maya/MIntArray.h>

// General Includes
//
#include <maya/MGlobal.h>
#include <maya/MPlug.h>
#include <maya/MDataBlock.h>
#include <maya/MDataHandle.h>
#include <maya/MIOStream.h>
#include <maya/MItGeometry.h>

// Macros
//
#define MCheckStatus(status,message)	\
	if( MStatus::kSuccess != status ) {	\
		cerr << message << "\n";		\
		return status;					\
	}

#include <SHelper.h>
#include <AHelper.h>
#include "H5FieldIn.h"
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/progress.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/format.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/iostreams/filter/gzip.hpp>

#include <sstream>
using namespace boost::filesystem;
using namespace std;
namespace io = boost::iostreams;

MTypeId AdaptiveFieldDeformer::id( 0xb855af ); 
MObject AdaptiveFieldDeformer::aframe;
MObject AdaptiveFieldDeformer::aminframe;
MObject AdaptiveFieldDeformer::amaxframe;
MObject AdaptiveFieldDeformer::acachename;
MObject AdaptiveFieldDeformer::areadloc;
MObject AdaptiveFieldDeformer::asubframe;
MultiPlaybackFile AdaptiveFieldDeformer::AvailableFieldFiles;

AdaptiveFieldDeformer::AdaptiveFieldDeformer()
{ m_lastFilename = ""; }

AdaptiveFieldDeformer::~AdaptiveFieldDeformer()
{}

MStatus AdaptiveFieldDeformer::deform(MDataBlock& block, MItGeometry& iter, const MMatrix& mat, unsigned int multiIndex)
{
	MStatus status = MS::kSuccess;
	
	MDataHandle envlData = block.inputValue(envelope,&status);
	float envl = envlData.asFloat();
	if(envl < .005f) 
	    return status;
	
	const MString filename =  block.inputValue( acachename ).asString();
	if(filename.length() < 3)
	    return status;
	
	double dtime = block.inputValue( aframe ).asDouble();
	int imin = block.inputValue( aminframe ).asInt();
	int imax = block.inputValue( amaxframe ).asInt();
	// const int isSubframe = block.inputValue(asubframe).asInt();

	int iframe = int(float(int(dtime * 1000 + 0.5))/1000.f);
	if(iframe < imin) iframe = imin;
	if(iframe > imax) iframe = imax;
	
	std::string substituded(filename.asChar());
	// EnvVar::replace(substituded);
	
	if(substituded != m_lastFilename) {
		openFieldFile(substituded);
		m_lastFilename = substituded;
	}
	
	APlaybackFile * file = AvailableFieldFiles.namedFile(substituded.c_str());
	if(file) {
		H5FieldIn * fieldf = (H5FieldIn *)file;
		//SampleFrame &sampler = AdaptiveFieldDeformer::H5Files.getFrameSampler(substituded.c_str());
		//sampler.calculateWeights(dtime);
		// MGlobal::displayInfo(MString("smp ")+ sampler.m_frames[0] + " " + sampler.m_frames[1] + " " + sampler.m_samples[0]  + " " + sampler.m_samples[1]  + " " + sampler.m_weights[0]  + " " + sampler.m_weights[1]);
		//if(isSubframe == 0)
		  //  sampler.m_weights[0] = 1.f;
		
		// float *p = new float[iter.count() * 3];
		
		// char hasData = readFrame(p, iter.count(), sampler.m_frames[0], sampler.m_samples[0]);
		
		//if(hasData) {
		//	if(sampler.m_weights[0] > 0.99f) {
		//		setP(envl, p, iter);
		//	}
		//	else {
		//		float *p1 = new float[iter.count() * 3];
		//		hasData = readFrame(p1, iter.count(), sampler.m_frames[1], sampler.m_samples[1]);
		//		if(hasData) {
		//			mixFrames(p, p1, iter.count(), sampler.m_weights[0], sampler.m_weights[1]);
		//		}
		//		setP(envl, p, iter);
					
		//		delete[] p1;
		//	}
		//}
		
		//delete[] p;
	}
	
	return status;
}

void* AdaptiveFieldDeformer::creator()
{
	return new AdaptiveFieldDeformer();
}

MStatus AdaptiveFieldDeformer::initialize()		
{
	MStatus				status;
	MFnNumericAttribute numAttr;
	MFnTypedAttribute tAttr;
	
	aframe = numAttr.create( "currentTime", "ct", MFnNumericData::kDouble, 1.0 );
	numAttr.setStorable(true);
	numAttr.setKeyable(true);
	addAttribute( aframe );
	
	aminframe = numAttr.create( "minFrame", "mnf", MFnNumericData::kInt, 1 );
	numAttr.setStorable(true);
	numAttr.setKeyable(true);
	addAttribute( aminframe );
	
	amaxframe = numAttr.create( "maxFrame", "mxf", MFnNumericData::kInt, 24 );
	numAttr.setStorable(true);
	numAttr.setKeyable(true);
	addAttribute( amaxframe );
	
	acachename = tAttr.create( "cachePath", "cp", MFnData::kString );
 	tAttr.setStorable(true);
	addAttribute( acachename );
	
	areadloc = numAttr.create( "readLocation", "rlc", MFnNumericData::kInt, 0 );
	numAttr.setStorable(true);
	numAttr.setKeyable(true);
	addAttribute( areadloc );
	
	asubframe = numAttr.create( "subframe", "sbf", MFnNumericData::kInt, 1 );
	numAttr.setStorable(true);
	numAttr.setKeyable(true);
	addAttribute(asubframe);

	// Set up a dependency between the input and the output.  This will cause
	// the output to be marked dirty when the input changes.  The output will
	// then be recomputed the next time the value of the output is requested.
	//
	
	attributeAffects(aframe, outputGeom);
	attributeAffects(asubframe, outputGeom);
	attributeAffects(acachename, outputGeom);

	return MS::kSuccess;

}

void AdaptiveFieldDeformer::setP(float env, float *p, MItGeometry& iter)
{
	for ( int i=0; !iter.isDone(); iter.next(), i++) {
		MPoint bent(p[i*3], p[i*3+1], p[i*3+2]);
		if(env > 0.99f) {
			iter.setPosition(bent);
		}
		else {
			MPoint ori = iter.position();
		
			MVector disp = bent - ori;
			bent = ori + disp * env;
			iter.setPosition(bent);
		}
	}
}

char AdaptiveFieldDeformer::readFrame(float *data, int count, int frame, int sample)
{	
	return 0;
}

void AdaptiveFieldDeformer::mixFrames(float *p, float *p1, int count, float weight0, float weight1)
{
	for ( int i=0; i < count * 3; i++)
		p[i] = p[i] * weight0 + p1[i] * weight1;
}

bool AdaptiveFieldDeformer::openFieldFile(const std::string & name)
{
	H5FieldIn * f = new H5FieldIn;
	if(!f->open(name)) {
		MGlobal::displayInfo(MString(" cannot open field file ")
								+ MString(name.c_str()));
		return false;
	}
	AvailableFieldFiles.addFile(f);
	return true;
}
//:~