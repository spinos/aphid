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

#ifdef WIN32
#include <gExtension.h>
#endif

MTypeId ExampViz::id( 0x95a20e );
MObject ExampViz::abboxminv;
MObject ExampViz::abboxmaxv;
MObject ExampViz::ancells;
MObject ExampViz::acellBuf;
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
	
		loadBoxes(block);
		
		MFnPluginData fnPluginData;
		MStatus status;
		MObject newDataObject = fnPluginData.create(ExampData::id, &status);
		
		ExampData * pData = (ExampData *) fnPluginData.data(&status);
		
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
	const BoundingBox & bbox = geomBox();
	MObject selfNode = thisMObject();
	MPlug rPlug(selfNode, adrawColorR);
	MPlug gPlug(selfNode, adrawColorG);
	MPlug bPlug(selfNode, adrawColorB);
	
	float * diffCol = diffuseMaterialColV();
	diffCol[0] = rPlug.asFloat();
	diffCol[1] = gPlug.asFloat();
	diffCol[2] = bPlug.asFloat();
	
	if(numBoxes() < 1) loadBoxes(selfNode);
	
	updateGeomBox(selfNode);
		
	view.beginGL();
	drawBoundingBox(&bbox);
	
	if ( style == M3dView::kFlatShaded || 
		    style == M3dView::kGouraudShaded ) {	
			
		glDepthFunc(GL_LEQUAL);
		glPushAttrib(GL_LIGHTING_BIT);
		glEnable(GL_LIGHTING);
		glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffCol );
			
		drawGrid();
		
		glDisable(GL_LIGHTING);
		glPopAttrib();
	} 
	else
		drawWireGrid();
	
	Matrix44F mat;
	mat.setFrontOrientation(Vector3F::YAxis);
	mat.scaleBy(geomSize() );
    mat.glMatrix(m_transBuf);
	
	drawCircle(m_transBuf);
	view.endGL();
}

bool ExampViz::isBounded() const
{ return true; }

MBoundingBox ExampViz::boundingBox() const
{   
	const BoundingBox & bbox = geomBox();
	
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
	
	attributeAffects(aradiusMult, outValue);
	attributeAffects(ancells, outValue);
	attributeAffects(acellBuf, outValue);
	attributeAffects(adrawColorR, outValue);
	attributeAffects(adrawColorG, outValue);
	attributeAffects(adrawColorB, outValue);
	attributeAffects(adrawColor, outValue);
	return MS::kSuccess;
}

MStatus ExampViz::connectionMade ( const MPlug & plug, const MPlug & otherPlug, bool asSrc )
{
	if(plug == outValue)
		AHelper::Info<MString>("connect", plug.name());
	return MPxLocatorNode::connectionMade (plug, otherPlug, asSrc );
}

MStatus ExampViz::connectionBroken ( const MPlug & plug, const MPlug & otherPlug, bool asSrc )
{
	if(plug == outValue)
		AHelper::Info<MString>("disconnect", plug.name());
	return MPxLocatorNode::connectionMade (plug, otherPlug, asSrc );
}

void ExampViz::voxelize(KdIntersection * tree)
{
	ExampVox::voxelize(tree);
	
	const BoundingBox bb = geomBox();
	
	MFnNumericData bbFn;
	MObject bbData = bbFn.create(MFnNumericData::k3Float);
	
	bbFn.setData(bb.data()[0], bb.data()[1], bb.data()[2]);
	MPlug bbmnPlug(thisMObject(), abboxminv);
	bbmnPlug.setValue(bbData);
	
	bbFn.setData(bb.data()[3], bb.data()[4], bb.data()[5]);
	MPlug bbmxPlug(thisMObject(), abboxmaxv);
	bbmxPlug.setValue(bbData);
	
	const unsigned n = numBoxes();
	MPlug ncellsPlug(thisMObject(), ancells);
	ncellsPlug.setInt((int)n);
	if(n < 1) return;
	
	float * src = boxCenterSizeF4();
	MPointArray pnts;
	pnts.setLength(n);
	unsigned i=0;
	for(;i<n;++i) {
		pnts[i] = MPoint(src[i*4], src[i*4+1], src[i*4+2], src[i*4+3]);
	}
	
	MFnPointArrayData pntFn;
	MObject opnt = pntFn.create(pnts);
	MPlug cellPlug(thisMObject(), acellBuf);
	cellPlug.setValue(opnt);
	
	AHelper::Info<unsigned>(" ExampViz generate n cells" ,n);
}

void ExampViz::updateGeomBox(MObject & node)
{
	MPlug radiusMultPlug(node, aradiusMult);
	float radiusScal = radiusMultPlug.asFloat();
	setGeomSizeMult(radiusScal);
	
	BoundingBox bb;
	
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
		AHelper::Info<unsigned>(" ExampViz error wrong cell data length", n );
		return;
	}
	
	n = numBoxes();
	setBoxes(pnts, n);
	
	AHelper::Info<unsigned>(" ExampViz load n cells", n );
}

void ExampViz::loadBoxes(MDataBlock & data)
{
	unsigned nc = data.inputValue(ancells).asInt();
	if(nc < 1) {
		AHelper::Info<unsigned>(" ExampViz error zero n cells", 0);
		return;
	}
	if(setNumBoxes(nc) ) {
	
		MDataHandle pntH = data.inputValue(acellBuf);
		MFnPointArrayData pntFn(pntH.data());
		MPointArray pnts = pntFn.array();
	
		unsigned n = pnts.length();
	
		if(n >= nc) {
			n = numBoxes();
			setBoxes(pnts, n);
		}
		else {
			AHelper::Info<unsigned>(" ExampViz error wrong cells length", pnts.length() );
		}
	}
	
	MDataHandle radiusMultH = data.inputValue(aradiusMult);
	float radiusScal = radiusMultH.asFloat();
	setGeomSizeMult(radiusScal);
	
	BoundingBox bb;
	
	MDataHandle bbminH = data.inputValue(abboxminv);
	MFloatVector& vmin = bbminH.asFloatVector();
	bb.setMin(vmin.x, vmin.y, vmin.z);
	
	MDataHandle bbmaxH = data.inputValue(abboxmaxv);
	MFloatVector& vmax = bbmaxH.asFloatVector();
	bb.setMax(vmax.x, vmax.y, vmax.z);

	setGeomBox(vmin.x, vmin.y, vmin.z,
				vmax.x, vmax.y, vmax.z);
	
	float * diffCol = diffuseMaterialColV();
	
	MFloatVector& c = data.inputValue(adrawColor).asFloatVector();
	diffCol[0] = c.x; diffCol[1] = c.y; diffCol[2] = c.z;
	
	AHelper::Info<unsigned>(" ExampViz update n cells", numBoxes() );
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
}