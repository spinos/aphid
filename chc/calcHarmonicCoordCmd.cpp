/*
 *  polyIntersection -p |group1|ball|ballShape -s |group2|sphere|sphereShape
 *
 */

#include "calcHarmonicCoordCmd.h"
#include <math.h>
#include <map>

#include <maya/MObject.h>
#include <maya/MGlobal.h>
#include <maya/MDagPath.h>
#include <maya/MDagPathArray.h>
#include <maya/MItSelectionList.h>
#include <maya/MSelectionList.h>
#include <maya/MIntArray.h>
#include <maya/MDoubleArray.h>
#include <maya/MObjectArray.h>
#include <maya/MItDependencyNodes.h>
#include <maya/MItGeometry.h>
#include <maya/MFnSkinCluster.h>
#include <maya/MItMeshVertex.h>
#include <maya/MFnSingleIndexedComponent.h>
#include <maya/MFnIntArrayData.h>
#include <maya/MIntArray.h>

#include <maya/MIOStream.h>
#include <boost/format.hpp>
#include <fstream>

#include "../opium/aSearchHelper.h"
#include <IntersectionGroup.h>

#include <MeshLaplacian.h>
#include <HarmonicCoord.h>

#define CheckError(stat,msg)		\
	if ( MS::kSuccess != stat ) {	\
		displayError(msg);			\
		continue;					\
	}


HarmonicCoordCmd::HarmonicCoordCmd()
{
}

HarmonicCoordCmd::~HarmonicCoordCmd() {}

void* HarmonicCoordCmd::creator()
{
    return new HarmonicCoordCmd;
}

bool HarmonicCoordCmd::isUndoable() const
{
    return false;
}

MStatus HarmonicCoordCmd::undoIt()
{
    return MS::kSuccess;
}

MStatus HarmonicCoordCmd::parseArgs( const MArgList& args )
//
// There is one mandatory flag: -f/-file <filename>
//
{
	MStatus     	stat;
	MString     	arg;
	const MString	anchorFlag			("-a");
	const MString	anchorFlagLong		("-anchor");
	const MString	valueFlag			("-v");
	const MString	valueFlagLong		("-value");
	const MString	helpFlag			("-h");
	const MString	helpFlagLong		("-help");
	
	m_anchorArribName = "asanchor";
	m_valueArribName = "asvalue";
	m_showHelp = false;

// Parse the arguments.
	for ( unsigned int i = 0; i < args.length(); i++ ) {
		arg = args.asString( i, &stat );
		if (!stat)              
			continue;
			
		if (i == args.length()-1) {
			args.get(i, m_meshName);
		}
				
		if ( arg == anchorFlag || arg == anchorFlagLong ) {
			if (i == args.length()-1) 
				continue;
			i++;
			args.get(i, m_anchorArribName);
		}
		else if ( arg == valueFlag || arg == valueFlagLong )
		{
			if (i == args.length()-1) 
				continue;
			i++;
			args.get(i, m_valueArribName);			
		}
		else if ( arg == helpFlag || arg == helpFlagLong ) {
			m_showHelp = true;
		}
	}

	return MS::kSuccess;
}


MStatus HarmonicCoordCmd::doIt( const MArgList& args )
{
	// parse args to get the file name from the command-line
	//
	MStatus stat = parseArgs(args);
	if (stat != MS::kSuccess) {
		return stat;
	}
	
	if(m_showHelp) {
		MGlobal::displayInfo("Calculate per-vertex value based on harmonic coordinate (d(v) = 0). Example:\n"
		"calcHarmCoord -a asanchor -v asvalue |mesh|meshShape\n"
		"asachor is double array attribute defines the vertices as anchors, asvalue is the double array attribute as input and output,\n"
		"last arguement is full path name of the mesh.");
		return MS::kSuccess;
	}
	
	ASearchHelper finder;
	
	MDagPath pMeshPri;
	if(!finder.dagByFullName(m_meshName.asChar(), pMeshPri)) {
		MGlobal::displayWarning(MString("cannot find object mesh: ") + m_meshName);
		return MS::kSuccess;
	}
	
	MFnMesh fmesh(pMeshPri.node(), &stat);
	if(!stat) {
		MGlobal::displayError(MString("object is not mesh ")+m_meshName);
		return MS::kFailure;
	}

	MPlug pasanchor = fmesh.findPlug(m_anchorArribName, &stat);
	if(!stat) {
		MGlobal::displayError(MString("cannot found attrib ")+m_anchorArribName);
		return MS::kFailure;
	}
	
	MPlug pvalue = fmesh.findPlug(m_valueArribName, &stat);
	if(!stat) {
		MGlobal::displayError(MString("cannot found attrib ")+m_valueArribName);
		return MS::kFailure;
	}
	
	MDoubleArray anchors, values;
	int anchorArraySize = doubleArraySize(pasanchor, anchors);
	if(anchorArraySize != fmesh.numVertices()) {
		MGlobal::displayError("anchor array has wrong size ");
		return MS::kFailure;
	}
	
	int valueArraySize = doubleArraySize(pvalue, values);
	if(valueArraySize != fmesh.numVertices()) {
		MGlobal::displayError("value array has wrong size ");
		return MS::kFailure;
	}
	
	MGlobal::displayInfo(MString("vertex count: ") + fmesh.numVertices());
	MeshLaplacian * mesh = new MeshLaplacian;
	
	unsigned nv = fmesh.numVertices();
	mesh->createVertices(nv);
	
	MPointArray vertexArray;
	fmesh.getPoints(vertexArray);
	for(unsigned i = 0; i < nv; i++) {
		mesh->setVertex(i, vertexArray[i].x, vertexArray[i].y, vertexArray[i].z);
	}
	
	MIntArray triangleCounts, triangleVertices;
	fmesh.getTriangles(triangleCounts, triangleVertices);
	
	unsigned ni = 0;
	for(unsigned i = 0; i < triangleCounts.length(); i++) {
		ni += triangleCounts[i];
	}
	mesh->createIndices(ni);
	
	for(unsigned i = 0; i < triangleVertices.length(); i+=3) {
		mesh->setTriangle(i / 3, triangleVertices[i], triangleVertices[i + 1], triangleVertices[i + 2]);
	}
	
	mesh->buildTopology();
	
	HarmonicCoord * harm = new HarmonicCoord;
	harm->setMesh(mesh);
	harm->addValue(1);
	
	return MS::kSuccess;
}


MStatus HarmonicCoordCmd::redoIt()
{
    clearResult();
	setResult( (int) 1);
    return MS::kSuccess;
}

int HarmonicCoordCmd::doubleArraySize(MPlug & plug, MDoubleArray & data) const
{
	MStatus stat;
	MObject oa = plug.asMObject();
	MFnDoubleArrayData fa(oa, &stat);
	if(!stat) MGlobal::displayError(MString("cannot build double array from attribue ") + plug.name());
	data = fa.array();
	return data.length();
}
