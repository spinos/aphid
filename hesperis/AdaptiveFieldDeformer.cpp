#include "AdaptiveFieldDeformer.h"
// General Includes
//
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
#include <IndexArray.h>
#include "H5FieldIn.h"
#include "AdaptiveField.h"
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
MObject AdaptiveFieldDeformer::asubframe;
MultiPlaybackFile AdaptiveFieldDeformer::AvailableFieldFiles;

AdaptiveFieldDeformer::AdaptiveFieldDeformer()
{ 
    m_lastFilename = ""; 
    m_field = NULL;
    m_cellIndices = new IndexArray;
    m_pieceCached = new IndexArray;
    m_elementOffset = 0;
}

AdaptiveFieldDeformer::~AdaptiveFieldDeformer()
{
    delete m_cellIndices;
    delete m_pieceCached;
}

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
	
    if(multiIndex == 0) m_elementOffset = 0;
    
	APlaybackFile * file = AvailableFieldFiles.namedFile(substituded.c_str());
	if(file) {
		H5FieldIn * fieldf = (H5FieldIn *)file;
		if(multiIndex == 0) {
// on frame change
			if(fieldf->currentFrame() != iframe) {
				fieldf->setCurrentFrame(iframe);
                MGlobal::displayInfo(MString("field deformer read frame ")+iframe);
				fieldf->readFrame();
			}
		}
       
        if(m_pieceCached->capacity() < (multiIndex + 1)) {
            m_pieceCached->expandBy(4);
        }
        
        if(m_cellIndices->capacity() < (m_elementOffset + iter.count())) {
            m_cellIndices->expandBy(4*iter.count());
        }
        
        unsigned * isCached = m_pieceCached->asIndex(multiIndex);
            
        m_cellIndices->setIndex(m_elementOffset);
        if(*isCached != 1) {
            AHelper::Info<unsigned>("field deformer build cell for geometry", multiIndex);
            cacheCellIndex(iter);
            *isCached = 1;
        }
        
// go back to begin
        m_cellIndices->setIndex(m_elementOffset);
        iter.reset();
        
        Vector3F * dP = (Vector3F *)m_field->namedData("dP");
        Vector3F samp;
        MPoint q;
        
        for (; !iter.isDone(); iter.next()) {
            MPoint ori = iter.position();
            samp.set(ori.x, ori.y, ori.z);
            samp += dP[*m_cellIndices->asIndex()];
            
            q.x = samp.x;
            q.y = samp.y;
            q.z = samp.z;
            
            iter.setPosition(q);
            m_cellIndices->next();
        }
        m_elementOffset += iter.count();
	}
	
	return status;
}

void AdaptiveFieldDeformer::cacheCellIndex(MItGeometry& iter)
{
    AHelper::Info<unsigned>(" piecemeal offset ", m_cellIndices->index());
    Vector3F samp;
    MPoint q;
    for (; !iter.isDone(); iter.next()) {
        MPoint ori = iter.position();
        samp.set(ori.x, ori.y, ori.z);
        m_field->putPInsideBound(samp);
        sdb::CellValue * c = m_field->locateCell(samp);
        if(c) *m_cellIndices->asIndex() = c->index;
        else *m_cellIndices->asIndex() = 0;
        
        //AHelper::Info<unsigned>("cell ", *m_cellIndices->asIndex());
            
        m_cellIndices->next();
    }
    
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
    const unsigned n = f->numFields();
    if(n < 1) {
        MGlobal::displayInfo("file has no field");
        return false;
    }
    
    m_field = (AdaptiveField *)f->fieldByIndex(0);

    MGlobal::displayInfo(MString("adaptive field n cells ") + m_field->numCells());
    f->setCurrentFrame(-999999);
	AvailableFieldFiles.addFile(f);
	return true;
}

void AdaptiveFieldDeformer::CloseAllFiles()
{ AvailableFieldFiles.cleanup();}
//:~