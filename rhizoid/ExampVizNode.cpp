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
#include <BoundingBox.h>
#include <AHelper.h>

#ifdef WIN32
#include <gExtension.h>
#endif

MTypeId ExampViz::id( 0x95a20e );
MObject ExampViz::abboxminv;
MObject ExampViz::abboxmaxv;
MObject ExampViz::ancells;
MObject ExampViz::acellBuf;
MObject ExampViz::adrawColor;
MObject ExampViz::outValue;

ExampViz::ExampViz()
{}

ExampViz::~ExampViz() 
{}

MStatus ExampViz::compute( const MPlug& plug, MDataBlock& block )
{
	if( plug == outValue ) {
	
		float result = 1.f;

		MDataHandle outputHandle = block.outputValue( outValue );
		outputHandle.set( result );
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
	MPlug colPlug(selfNode, adrawColor);
	MObject col;
	colPlug.getValue(col);
	MFnNumericData colFn(col);
	float * diffCol = diffuseMaterialColV();
	colFn.getData(diffCol[0], diffCol[1], diffCol[2]);
	
	if(numBoxes() < 1) loadBoxes(selfNode);
		
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
	
	adrawColor = numFn.createColor( "dspColor", "dspc" );
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setUsedAsColor(true);
	numFn.setDefault(0.47f, 0.46f, 0.45f);
	addAttribute(adrawColor);
	
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
	
	outValue = numFn.create( "outValue", "ov", MFnNumericData::kFloat );
	numFn.setStorable(false);
	numFn.setWritable(false);
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
	
	const unsigned n = pnts.length();
	if(n != numBoxes() ) {
		AHelper::Info<unsigned>(" ExampViz error wrong cell data length", n );
		return;
	}
	
	float * dst = boxCenterSizeF4();
	unsigned i=0;
	for(;i<n;++i) {
		const MPoint & p = pnts[i];
		dst[i*4] = p.x;
		dst[i*4+1] = p.y;
		dst[i*4+2] = p.z;
		dst[i*4+3] = p.w;
	}
	
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
	
	setGeomBox(bb);
}