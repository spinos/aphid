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
#include <maya/MFnPluginData.h>
#include <maya/MFnVectorArrayData.h>
#include <maya/MFnMatrixAttribute.h>
#include <math/BoundingBox.h>
#include <AHelper.h>
#include <ExampData.h>
#include <math/linearMath.h>
#include <mama/MeshHelper.h>

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
MObject ExampViz::adrawDopSizeX;
MObject ExampViz::adrawDopSizeY;
MObject ExampViz::adrawDopSizeZ;
MObject ExampViz::adrawDopSize;
MObject ExampViz::aradiusMult;
MObject ExampViz::aininstspace;
MObject ExampViz::outValue;

using namespace aphid;

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
		
		BoundingBox bb;
		
		MDataHandle bbminH = block.inputValue(abboxminv);
		MFloatVector& vmin = bbminH.asFloatVector();
		bb.setMin(vmin.x, vmin.y, vmin.z);
		
		MDataHandle bbmaxH = block.inputValue(abboxmaxv);
		MFloatVector& vmax = bbmaxH.asFloatVector();
		bb.setMax(vmax.x, vmax.y, vmax.z);

		setGeomBox(vmin.x, vmin.y, vmin.z,
				vmax.x, vmax.y, vmax.z);
		
		MDataHandle drszx = block.inputValue(adrawDopSizeX);
		MDataHandle drszy = block.inputValue(adrawDopSizeY);
		MDataHandle drszz = block.inputValue(adrawDopSizeZ);
		setDopSize(drszx.asFloat(), drszy.asFloat(), drszz.asFloat() );
	
		float * diffCol = diffuseMaterialColV();
		
		MFloatVector c = block.inputValue(adrawColor).asFloatVector();
		diffCol[0] = c.x; diffCol[1] = c.y; diffCol[2] = c.z;
		
		if(!loadDops(block) ) {
			AHelper::Info<MString>(" ERROR ExampViz has no draw data", MFnDependencyNode(thisMObject() ).name() );
		}
		
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
	MObject selfNode = thisMObject();
	updateGeomBox(selfNode);
	
	MPlug rPlug(selfNode, adrawColorR);
	MPlug gPlug(selfNode, adrawColorG);
	MPlug bPlug(selfNode, adrawColorB);
	
	float * diffCol = diffuseMaterialColV();
	diffCol[0] = rPlug.asFloat();
	diffCol[1] = gPlug.asFloat();
	diffCol[2] = bPlug.asFloat();
	
	MPlug szxp(selfNode, adrawDopSizeX);
	MPlug szyp(selfNode, adrawDopSizeY);
	MPlug szzp(selfNode, adrawDopSizeZ);
	setDopSize(szxp.asFloat(), szyp.asFloat(), szzp.asFloat() );
	
/// load dop first, then box
	bool stat = dopBufLength() > 0;
	if(!stat) {
		stat = loadDOPs(selfNode);
	}
	if(!stat) {
		AHelper::Info<MString>(" ERROR ExampViz has no draw data", MFnDependencyNode(selfNode).name() );
		return;
	}
	
	view.beginGL();
	
	const BoundingBox & bbox = geomBox();
	drawBoundingBox(&bbox);
	
	//if ( style == M3dView::kFlatShaded || 
	//	    style == M3dView::kGouraudShaded ) {
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glEnableClientState(GL_VERTEX_ARRAY);
	drawAWireDop();
	glColor3fv(diffCol);
	drawWiredTriangles();
	glDisableClientState(GL_VERTEX_ARRAY);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	//} 
	
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
	
	adrawDopSizeX = numFn.create( "dspDopX", "ddpx", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.9f);
	numFn.setMin(0.1f);
	numFn.setMax(1.f);
	addAttribute(adrawDopSizeX);
	
	adrawDopSizeY = numFn.create( "dspDopY", "ddpy", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.9f);
	numFn.setMin(0.1f);
	numFn.setMax(1.f);
	addAttribute(adrawDopSizeY);
	
	adrawDopSizeZ = numFn.create( "dspDopZ", "ddpz", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.9f);
	numFn.setMin(0.1f);
	numFn.setMax(1.f);
	addAttribute(adrawDopSizeZ);
	
	adrawDopSize = numFn.create( "dspDop", "ddps", adrawDopSizeX, adrawDopSizeY, adrawDopSizeZ );
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setUsedAsColor(true);
	numFn.setDefault(0.9f, 0.9f, 0.9f);
	addAttribute(adrawDopSize);
	
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
	
	MFnMatrixAttribute matAttr;
	aininstspace = matAttr.create("instanceSpace", "sinst", MFnMatrixAttribute::kDouble);
	matAttr.setStorable(false);
	matAttr.setWritable(true);
	matAttr.setConnectable(true);
    matAttr.setArray(true);
    matAttr.setDisconnectBehavior(MFnAttribute::kDelete);
	addAttribute( aininstspace );
	
	attributeAffects(aradiusMult, outValue);
	attributeAffects(ancells, outValue);
	attributeAffects(acellBuf, outValue);
	attributeAffects(adoplen, outValue);
	attributeAffects(adopPBuf, outValue);
	attributeAffects(adopNBuf, outValue);
	attributeAffects(adrawColorR, outValue);
	attributeAffects(adrawColorG, outValue);
	attributeAffects(adrawColorB, outValue);
	attributeAffects(adrawDopSizeX, outValue);
	attributeAffects(adrawDopSizeY, outValue);
	attributeAffects(adrawDopSizeZ, outValue);
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

void ExampViz::setTriangleMesh(const DenseMatrix<float> & pnts,
						const MIntArray & triangleVertices,
						const BoundingBox & bbox)
{
	MFnNumericData bbFn;
	MObject bbData = bbFn.create(MFnNumericData::k3Float);
	
	bbFn.setData(bbox.getMin(0), bbox.getMin(1), bbox.getMin(2));
	MPlug bbmnPlug(thisMObject(), abboxminv);
	bbmnPlug.setValue(bbData);
	
	bbFn.setData(bbox.getMax(0), bbox.getMax(1), bbox.getMax(2));
	MPlug bbmxPlug(thisMObject(), abboxmaxv);
	bbmxPlug.setValue(bbData);
	
	const int nind = triangleVertices.length();
	const int & np = pnts.numCols();
	MPlug dopLenPlug(thisMObject(), adoplen);
	dopLenPlug.setInt(nind);
	
	MVectorArray vecp; 
	MeshHelper::ScatterTriangleVerticesPosition(vecp,
						pnts.column(0), np,
						triangleVertices, nind);
						
	MVectorArray vecn; 
	MeshHelper::CalculateTriangleVerticesNormal(vecn,
						pnts.column(0), np,
						triangleVertices, nind);
						
	MFnVectorArrayData vecFn;
	MObject opnt = vecFn.create(vecp);
	MPlug doppPlug(thisMObject(), adopPBuf);
	doppPlug.setValue(opnt);
	
	MObject onor = vecFn.create(vecn);
	MPlug dopnPlug(thisMObject(), adopNBuf);
	dopnPlug.setValue(onor);
	
	AHelper::Info<unsigned>(" ExampViz load n points", np );
	AHelper::Info<unsigned>(" n triangle vertex", nind );
}

void ExampViz::voxelize2(sdb::VectorArray<cvx::Triangle> * tri,
							const BoundingBox & bbox)
{
	ExampVox::voxelize2(tri, bbox);
	
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
	const Vector3F * ps = dopPositionR();
	const Vector3F * ns = dopNormalR();
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
	
	AHelper::Info<int>("reduced draw n triangle ", dopBufLength() / 3 );
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
	
	AHelper::Info<unsigned>(" ExampViz load n boxes", n );
}

bool ExampViz::loadBoxes(MDataBlock & data)
{
	unsigned nc = data.inputValue(ancells).asInt();
	if(nc < 1) {
		AHelper::Info<unsigned>(" ExampViz error zero n cells", 0);
		return false;
	}

	MDataHandle pntH = data.inputValue(acellBuf);
	MFnPointArrayData pntFn(pntH.data());
	MPointArray pnts = pntFn.array();

	unsigned n = pnts.length();
	
	if(n < nc) {
		AHelper::Info<unsigned>(" ExampViz error wrong cells length", pnts.length() );
		return false;
	}
	
	setNumBoxes(nc);
	n = numBoxes();
	setBoxes(pnts, n);
	
	AHelper::Info<unsigned>(" ExampViz update n boxes", numBoxes() );
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
		AHelper::Info<int>(" ExampViz error zero n dops", n);
		return false;
	}
	
	MDataHandle pntH = data.inputValue(adopPBuf);
	MFnVectorArrayData pntFn(pntH.data());
	MVectorArray pnts = pntFn.array();
	
	if(pnts.length() < n) {
		AHelper::Info<unsigned>(" ExampViz error wrong dop position length", pnts.length() );
		return false;
	}
	
	MDataHandle norH = data.inputValue(adopNBuf);
	MFnVectorArrayData norFn(norH.data());
	MVectorArray nors = norFn.array();
	
	if(nors.length() < n) {
		AHelper::Info<unsigned>(" ExampViz error wrong dop normal length", pnts.length() );
		return false;
	}
	
	setDOPDrawBufLen(n);
	
	Vector3F * ps = dopPositionR();
	Vector3F * ns = dopNormalR();
	for(int i=0; i<n; ++i) {
		ps[i].set(pnts[i].x, pnts[i].y, pnts[i].z);
		ns[i].set(nors[i].x, nors[i].y, nors[i].z);
	}
	
	AHelper::Info<int>(" ExampViz load dop buf length", dopBufLength() );
	buildBounding8Dop(geomBox() );
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
		AHelper::Info<unsigned>(" ExampViz error wrong dop position length", np );
		return false;
	}
	
	MPlug dopnPlug(node, adopNBuf);
	MObject dopnObj;
	dopnPlug.getValue(dopnObj);
	
	MFnVectorArrayData norFn(dopnObj);
	MVectorArray nors = norFn.array();
	
	unsigned nn = nors.length();
	if(nn < n ) {
		AHelper::Info<unsigned>(" ExampViz error wrong dop normal length", nn );
		return false;
	}

	setDOPDrawBufLen(n);
	
	Vector3F * ps = dopPositionR();
	Vector3F * ns = dopNormalR();
	for(int i=0; i<n; ++i) {
		ps[i].set(pnts[i].x, pnts[i].y, pnts[i].z);
		ns[i].set(nors[i].x, nors[i].y, nors[i].z);
	}
	
	AHelper::Info<unsigned>(" ExampViz load n dops", n );
	buildBounding8Dop(geomBox() );
	return true;
}
//:~