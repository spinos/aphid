/*
 *  ExampVizNode.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 2/5/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "ExampVizNode.h"
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnPointArrayData.h>
#include <maya/MFloatVector.h>
#include <BoundingBox.h>
#include <AHelper.h>
#include <ExampData.h>
#include <KdTree.h>

#ifdef WIN32
#include <gExtension.h>
#endif

MTypeId ExampViz::id( 0x95a20e );
MObject ExampViz::abboxminv;
MObject ExampViz::abboxmaxv;
MObject ExampViz::ancells;
MObject ExampViz::acellBuf;
MObject ExampViz::adoplen;
MObject ExampViz::adopPBuf;
MObject ExampViz::adopNBuf;
MObject ExampViz::adrawColor;
MObject ExampViz::adrawColorR;
MObject ExampViz::adrawColorG;
MObject ExampViz::adrawColorB;
MObject ExampViz::aradiusMult;
MObject ExampViz::outValue;

ExampViz::ExampViz()
{}

ExampViz::~ExampViz() 
{}

MStatus ExampViz::compute( const MPlug& plug, MDataBlock& block )
{
	if( plug == outValue ) {
		
		MDataHandle radiusMultH = block.inputValue(aradiusMult);
		float radiusScal = radiusMultH.asFloat();
		setGeomSizeMult(radiusScal);
		
		aphid::BoundingBox bb;
		
		MDataHandle bbminH = block.inputValue(abboxminv);
		MFloatVector& vmin = bbminH.asFloatVector();
		bb.setMin(vmin.x, vmin.y, vmin.z);
		
		MDataHandle bbmaxH = block.inputValue(abboxmaxv);
		MFloatVector& vmax = bbmaxH.asFloatVector();
		bb.setMax(vmax.x, vmax.y, vmax.z);

		setGeomBox(vmin.x, vmin.y, vmin.z,
				vmax.x, vmax.y, vmax.z);
	
		float * diffCol = diffuseMaterialColV();
		
		MFloatVector& c = block.inputValue(adrawColor).asFloatVector();
		diffCol[0] = c.x; diffCol[1] = c.y; diffCol[2] = c.z;
		
/// dop first, then box
		if(!loadDops(block) )
			loadBoxes(block);
		
		MFnPluginData fnPluginData;
		MStatus status;
		MObject newDataObject = fnPluginData.create(aphid::ExampData::id, &status);
		
		aphid::ExampData * pData = (aphid::ExampData *) fnPluginData.data(&status);
		
		if(pData) pData->setDesc(this);

		MDataHandle outputHandle = block.outputValue( outValue );
		outputHandle.set( pData );
		block.setClean(plug);
    }

	return MS::kSuccess;	
}

void ExampViz::draw( M3dView & view, const MDagPath & path, 
							 M3dView::DisplayStyle style,
							 M3dView::DisplayStatus status )
{
	const aphid::BoundingBox & bbox = geomBox();
	MObject selfNode = thisMObject();
	MPlug rPlug(selfNode, adrawColorR);
	MPlug gPlug(selfNode, adrawColorG);
	MPlug bPlug(selfNode, adrawColorB);
	
	float * diffCol = diffuseMaterialColV();
	diffCol[0] = rPlug.asFloat();
	diffCol[1] = gPlug.asFloat();
	diffCol[2] = bPlug.asFloat();
	
/// load dop first, then box
	bool stat = dopBufLength() > 0;
	if(!stat) stat = loadDOPs(selfNode);
	if(!stat) loadBoxes(selfNode);
	
	updateGeomBox(selfNode);
		
	view.beginGL();
	drawBoundingBox(&bbox);
	
	//if ( style == M3dView::kFlatShaded || 
	//	    style == M3dView::kGouraudShaded ) {	
			
		glDepthFunc(GL_LEQUAL);
		glPushAttrib(GL_LIGHTING_BIT);
		glEnable(GL_LIGHTING);
		glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffCol );
			
		drawDop();
		
		glDisable(GL_LIGHTING);
		glPopAttrib();
	//} 
	
	aphid::Matrix44F mat;
	mat.setFrontOrientation(aphid::Vector3F::YAxis);
	mat.scaleBy(geomSize() );
    mat.glMatrix(m_transBuf);
	
	drawCircle(m_transBuf);
	view.endGL();
}

bool ExampViz::isBounded() const
{ return true; }

MBoundingBox ExampViz::boundingBox() const
{   
	const aphid::BoundingBox & bbox = geomBox();
	
	MPoint corner1(bbox.m_data[0], bbox.m_data[1], bbox.m_data[2]);
	MPoint corner2(bbox.m_data[3], bbox.m_data[4], bbox.m_data[5]);

	return MBoundingBox( corner1, corner2 );
}

void* ExampViz::creator()
{
	return new ExampViz();
}

MStatus ExampViz::initialize()
{ 
	MFnNumericAttribute numFn;
	MFnTypedAttribute typedFn;
	
	MStatus			 stat;
	
	adrawColorR = numFn.create( "dspColorR", "dspr", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.47f);
	addAttribute(adrawColorR);
	
	adrawColorG = numFn.create( "dspColorG", "dspg", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.46f);
	addAttribute(adrawColorG);
	
	adrawColorB = numFn.create( "dspColorB", "dspb", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.45f);
	addAttribute(adrawColorB);
	
	adrawColor = numFn.create( "dspColor", "dspc", adrawColorR, adrawColorG, adrawColorB );
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setUsedAsColor(true);
	numFn.setDefault(0.47f, 0.46f, 0.45f);
	addAttribute(adrawColor);
	
	aradiusMult = numFn.create( "radiusMultiplier", "rml", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(1.f);
	numFn.setMin(.05f);
	addAttribute(aradiusMult);
	
	ancells = numFn.create( "numCells", "ncl", MFnNumericData::kInt, 0 );
	numFn.setStorable(true);
	addAttribute(ancells);
	
	abboxminv = numFn.create( "BBoxMin", "bbxmn", MFnNumericData::k3Float );
	numFn.setStorable(true);
	numFn.setDefault(-1.f, -1.f, -1.f);
	addAttribute(abboxminv);
	
	abboxmaxv = numFn.create( "BBoxMax", "bbxmx", MFnNumericData::k3Float );
	numFn.setStorable(true);
	numFn.setDefault(1.f, 1.f, 1.f);
	addAttribute(abboxmaxv);
	
	outValue = typedFn.create( "outValue", "ov", MFnData::kPlugin );
	typedFn.setStorable(false);
	typedFn.setWritable(false);
	addAttribute(outValue);
	
	MPointArray defaultPntArray;
	MFnPointArrayData pntArrayDataFn;
	pntArrayDataFn.create( defaultPntArray );
	acellBuf = typedFn.create( "cellData", "cld",
											MFnData::kPointArray,
											pntArrayDataFn.object(),
											&stat );
    typedFn.setStorable(true);
	addAttribute(acellBuf);
	
	adoplen = numFn.create( "dopLen", "dpl", MFnNumericData::kInt, 0 );
	numFn.setStorable(true);
	addAttribute(adoplen);
	
	MVectorArray defaultVecArray;
	MFnVectorArrayData vecArrayDataFn;
	vecArrayDataFn.create( defaultVecArray );
	adopPBuf = typedFn.create( "dopPBuf", "dpp",
											MFnData::kVectorArray,
											vecArrayDataFn.object(),
											&stat );
    typedFn.setStorable(true);
	addAttribute(adopPBuf);
	
	adopNBuf = typedFn.create( "dopNBuf", "dpn",
											MFnData::kVectorArray,
											vecArrayDataFn.object(),
											&stat );
    typedFn.setStorable(true);
	addAttribute(adopNBuf);
	
	attributeAffects(aradiusMult, outValue);
	attributeAffects(ancells, outValue);
	attributeAffects(acellBuf, outValue);
	attributeAffects(adoplen, outValue);
	attributeAffects(adopPBuf, outValue);
	attributeAffects(adopNBuf, outValue);
	attributeAffects(adrawColorR, outValue);
	attributeAffects(adrawColorG, outValue);
	attributeAffects(adrawColorB, outValue);
	attributeAffects(adrawColor, outValue);
	return MS::kSuccess;
}

MStatus ExampViz::connectionMade ( const MPlug & plug, const MPlug & otherPlug, bool asSrc )
{
	if(plug == outValue)
		aphid::AHelper::Info<MString>("connect", plug.name());
	return MPxLocatorNode::connectionMade (plug, otherPlug, asSrc );
}

MStatus ExampViz::connectionBroken ( const MPlug & plug, const MPlug & otherPlug, bool asSrc )
{
	if(plug == outValue)
		aphid::AHelper::Info<MString>("disconnect", plug.name());
	return MPxLocatorNode::connectionMade (plug, otherPlug, asSrc );
}

void ExampViz::voxelize2(aphid::sdb::VectorArray<aphid::cvx::Triangle> * tri,
							const aphid::BoundingBox & bbox)
{
	aphid::ExampVox::voxelize2(tri, bbox);
	
	MFnNumericData bbFn;
	MObject bbData = bbFn.create(MFnNumericData::k3Float);
	
	bbFn.setData(bbox.data()[0], bbox.data()[1], bbox.data()[2]);
	MPlug bbmnPlug(thisMObject(), abboxminv);
	bbmnPlug.setValue(bbData);
	
	bbFn.setData(bbox.data()[3], bbox.data()[4], bbox.data()[5]);
	MPlug bbmxPlug(thisMObject(), abboxmaxv);
	bbmxPlug.setValue(bbData);
	
	const int n = dopBufLength();
	MPlug dopLenPlug(thisMObject(), adoplen);
	dopLenPlug.setInt(n);
	if(n < 1) return;
	
	MVectorArray dopp; dopp.setLength(n);
	MVectorArray dopn; dopn.setLength(n);
	const aphid::Vector3F * ps = dopPositionR();
	const aphid::Vector3F * ns = dopNormalR();
	for(int i=0; i<n; ++i) {
		dopp[i] = MVector(ps[i].x, ps[i].y, ps[i].z);
		dopn[i] = MVector(ns[i].x, ns[i].y, ns[i].z);
	}
	
	MFnVectorArrayData vecFn;
	MObject opnt = vecFn.create(dopp);
	MPlug doppPlug(thisMObject(), adopPBuf);
	doppPlug.setValue(opnt);
	
	MObject onor = vecFn.create(dopn);
	MPlug dopnPlug(thisMObject(), adopNBuf);
	dopnPlug.setValue(onor);
	
	aphid::AHelper::Info<int>("reduced draw n triangle ", dopBufLength() / 3 );
}

void ExampViz::updateGeomBox(MObject & node)
{
	MPlug radiusMultPlug(node, aradiusMult);
	float radiusScal = radiusMultPlug.asFloat();
	setGeomSizeMult(radiusScal);
	
	aphid::BoundingBox bb;
	
	MObject bbmn;
	MPlug bbmnPlug(node, abboxminv);
	bbmnPlug.getValue(bbmn);
	MFnNumericData bbmnFn(bbmn);
	bbmnFn.getData3Float(bb.m_data[0], bb.m_data[1], bb.m_data[2]);
	
	MObject bbmx;
	MPlug bbmxPlug(node, abboxmaxv);
	bbmxPlug.getValue(bbmx);
	MFnNumericData bbmxFn(bbmx);
	bbmxFn.getData3Float(bb.m_data[3], bb.m_data[4], bb.m_data[5]);
	
	setGeomBox(bb.m_data[0], bb.m_data[1], bb.m_data[2],
				bb.m_data[3], bb.m_data[4], bb.m_data[5]);
				
}

void ExampViz::loadBoxes(MObject & node)
{	
	MPlug ncellsPlug(node, ancells);
	int nc = ncellsPlug.asInt();
	if(nc<1) return;
	setNumBoxes(nc);
	
	MPlug cellPlug(node, acellBuf);
	MObject cellObj;
	cellPlug.getValue(cellObj);
	
	MFnPointArrayData pntFn(cellObj);
	MPointArray pnts = pntFn.array();
	
	unsigned n = pnts.length();
	if(n < numBoxes() ) {
		aphid::AHelper::Info<unsigned>(" ExampViz error wrong cell data length", n );
		return;
	}
	
	n = numBoxes();
	setBoxes(pnts, n);
	
	aphid::AHelper::Info<unsigned>(" ExampViz load n boxes", n );
}

bool ExampViz::loadBoxes(MDataBlock & data)
{
	unsigned nc = data.inputValue(ancells).asInt();
	if(nc < 1) {
		aphid::AHelper::Info<unsigned>(" ExampViz error zero n cells", 0);
		return false;
	}

	MDataHandle pntH = data.inputValue(acellBuf);
	MFnPointArrayData pntFn(pntH.data());
	MPointArray pnts = pntFn.array();

	unsigned n = pnts.length();
	
	if(n < nc) {
		aphid::AHelper::Info<unsigned>(" ExampViz error wrong cells length", pnts.length() );
		return false;
	}
	
	setNumBoxes(nc);
	n = numBoxes();
	setBoxes(pnts, n);
	
	aphid::AHelper::Info<unsigned>(" ExampViz update n boxes", numBoxes() );
	return true;
}

void ExampViz::setBoxes(const MPointArray & src, const unsigned & num)
{
	float * dst = boxCenterSizeF4();
	unsigned i=0;
	for(;i<num;++i) {
		const MPoint & p = src[i];
		dst[i*4] = p.x;
		dst[i*4+1] = p.y;
		dst[i*4+2] = p.z;
		dst[i*4+3] = p.w;
	}
	buildBoxDrawBuf();
}

bool ExampViz::loadDops(MDataBlock & data)
{
	int n = data.inputValue(adoplen).asInt();
	if(n < 1) {
		aphid::AHelper::Info<int>(" ExampViz error zero n dops", n);
		return false;
	}
	
	MDataHandle pntH = data.inputValue(adopPBuf);
	MFnVectorArrayData pntFn(pntH.data());
	MVectorArray pnts = pntFn.array();
	
	if(pnts.length() < n) {
		aphid::AHelper::Info<unsigned>(" ExampViz error wrong dop position length", pnts.length() );
		return false;
	}
	
	MDataHandle norH = data.inputValue(adopNBuf);
	MFnVectorArrayData norFn(norH.data());
	MVectorArray nors = norFn.array();
	
	if(nors.length() < n) {
		aphid::AHelper::Info<unsigned>(" ExampViz error wrong dop normal length", pnts.length() );
		return false;
	}
	
	setDOPDrawBufLen(n);
	
	aphid::Vector3F * ps = dopPositionR();
	aphid::Vector3F * ns = dopNormalR();
	for(int i=0; i<n; ++i) {
		ps[i].set(pnts[i].x, pnts[i].y, pnts[i].z);
		ns[i].set(nors[i].x, nors[i].y, nors[i].z);
	}
	
	aphid::AHelper::Info<int>(" ExampViz load dop buf length", dopBufLength() );
		
	return true;
}

bool ExampViz::loadDOPs(MObject & node)
{
	MPlug doplenPlug(node, adoplen);
	int n = doplenPlug.asInt();
	if(n<1) return false;
	
	MPlug doppPlug(node, adopPBuf);
	MObject doppObj;
	doppPlug.getValue(doppObj);
	
	MFnVectorArrayData pntFn(doppObj);
	MVectorArray pnts = pntFn.array();
	
	unsigned np = pnts.length();
	if(np < n ) {
		aphid::AHelper::Info<unsigned>(" ExampViz error wrong dop position length", np );
		return false;
	}
	
	MPlug dopnPlug(node, adopNBuf);
	MObject dopnObj;
	dopnPlug.getValue(dopnObj);
	
	MFnVectorArrayData norFn(dopnObj);
	MVectorArray nors = norFn.array();
	
	unsigned nn = nors.length();
	if(nn < n ) {
		aphid::AHelper::Info<unsigned>(" ExampViz error wrong dop normal length", nn );
		return false;
	}

	setDOPDrawBufLen(n);
	
	aphid::Vector3F * ps = dopPositionR();
	aphid::Vector3F * ns = dopNormalR();
	for(int i=0; i<n; ++i) {
		ps[i].set(pnts[i].x, pnts[i].y, pnts[i].z);
		ns[i].set(nors[i].x, nors[i].y, nors[i].z);
	}
	
	aphid::AHelper::Info<unsigned>(" ExampViz load n dops", n );
	
	return true;
}
//:~