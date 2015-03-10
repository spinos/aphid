#include "ExrImgLoader.h"
#include <maya/MPlug.h>
#include <maya/MDataBlock.h>
#include <maya/MDataHandle.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnUnitAttribute.h>
#include <maya/MTime.h>
#include <maya/MVector.h>
#include <maya/MMatrix.h>
#include <maya/MFnMatrixData.h>
#include <maya/MFnPluginData.h>
#include <maya/MGlobal.h>
#include <maya/MFnIntArrayData.h>
#include <maya/MFnVectorArrayData.h>
#include <SHelper.h>
MTypeId     ExrImgLoader::id(0x22c78c48);
MObject     ExrImgLoader::adepthImageName;
MObject     ExrImgLoader::aframeNumber;
MObject     ExrImgLoader::aframePadding;
MObject     ExrImgLoader::aenableMultiFrames;
MObject     ExrImgLoader::aoutval;

ExrImgLoader::ExrImgLoader()
{
    m_colBuf = new CUDABuffer;
    m_depBuf = new CUDABuffer;
    m_exr = 0;
	_pDesc = new ExrImgData::DataDesc;
	_pDesc->_isValid = 0;
}

ExrImgLoader::~ExrImgLoader() 
{
	if(_pDesc) delete _pDesc;
	if(m_exr) delete m_exr;
	delete m_colBuf;
	delete m_depBuf;
}

MStatus ExrImgLoader::compute( const MPlug& plug, MDataBlock& block )
{
	if(plug == aoutval) {
	    MString filename = block.inputValue(adepthImageName).asString();
        int frame = block.inputValue(aframeNumber).asInt();
		int padding = block.inputValue(aframePadding).asInt();
		bool enableSequence = block.inputValue(aenableMultiFrames).asBool();

		MStatus status;
		MGlobal::displayInfo("heather loader compute");
		preLoadImage(_pDesc, filename.asChar(), frame, padding, enableSequence);
		
		MFnPluginData fnPluginData;
		MObject newDataObject = fnPluginData.create(ExrImgData::id, &status);
		
		ExrImgData * pData = (ExrImgData *) fnPluginData.data(&status);
		
		if(pData) pData->setDesc(_pDesc);
	
		MGlobal::displayInfo("update exr image data desc");
		
		MDataHandle outDescHandle = block.outputValue(aoutval);
		status = outDescHandle.set(pData);

		block.setClean(plug);
		
		return MS::kSuccess;
	}
	
	return MS::kUnknownParameter;
}

void* ExrImgLoader::creator()
{
	return new ExrImgLoader();
}

MStatus ExrImgLoader::initialize()		
{
    MFnTypedAttribute typedAttr;
	MFnNumericAttribute numAttr;
	MStatus			 stat;
	
	adepthImageName = typedAttr.create( "depthImage", "dmg", MFnData::kString );
 	typedAttr.setStorable(true);
	addAttribute(adepthImageName);
	
	aenableMultiFrames = numAttr.create( "useImageSequence", "uis", MFnNumericData::kBoolean );
	numAttr.setStorable(true);
	addAttribute(aenableMultiFrames);
	
	aframeNumber = numAttr.create( "frameNumber", "fnb", MFnNumericData::kInt );
	numAttr.setStorable(true);
	numAttr.setKeyable(true);
	addAttribute(aframeNumber);
	
	aframePadding = numAttr.create( "framePadding", "fpd", MFnNumericData::kInt );
	numAttr.setDefault(0);
	numAttr.setStorable(true);
	addAttribute(aframePadding);
	
	aoutval = typedAttr.create("outValue", "ov", MFnData::kPlugin);
	typedAttr.setStorable(false);
	typedAttr.setConnectable(true);
	addAttribute(aoutval);
    
	attributeAffects(adepthImageName, aoutval);
	attributeAffects(aenableMultiFrames, aoutval);
	attributeAffects(aframeNumber, aoutval);

	return MS::kSuccess;
}

void ExrImgLoader::preLoadImage(ExrImgData::DataDesc * dst, const char * name, int frame, int padding, bool useImageSequence)
{
    std::string fileName(name);
    if(fileName.size() < 3) {
        dst->_isValid = 0;
        return;
    }
	
	if(useImageSequence)
		SHelper::changeFrameNumber(fileName, frame, padding);
	
	MGlobal::displayInfo(MString("heather loader loads image ")+fileName.c_str());
		
	if(!m_exr) m_exr = new ZEXRImage(fileName.c_str(), false);
	else m_exr->open(fileName.c_str());
	
    if(!m_exr->isOpened()) {
		MGlobal::displayInfo(MString("cannot open image ") + fileName.c_str());
		dst->_isValid = 0;
		return;
	}
	
	if(m_exr->fileName() != fileName) {
		MGlobal::displayInfo(MString("cannot open image ") + fileName.c_str());
		dst->_isValid = 0;
		return;
	}

    if(!m_exr->isRGBAZ()) {
        MGlobal::displayWarning(MString("image is not RGBAZ format.") + fileName.c_str());
        dst->_isValid = 0;
        return;
    }
    
    const unsigned numPix = m_exr->getWidth() * m_exr->getHeight();
    m_colBuf->create(numPix * 4 * 2);
    m_depBuf->create(numPix * 4);
    
    m_colBuf->hostToDevice(m_exr->_pixels);
    m_depBuf->hostToDevice(m_exr->m_zData);
    dst->_img = m_exr;
    dst->_colorBuf = m_colBuf;
    dst->_depthBuf = m_depBuf;
    dst->_isValid = 1;
}
//~: