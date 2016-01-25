/*
 *  UntangleCmd.cpp
 *  manuka
 *
 *  Created by jian zhang on 1/22/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "UntangleCmd.h"

#include <maya/MArgDatabase.h>
#include <maya/MArgList.h>
#include <maya/MPointArray.h>
#include <maya/MDagModifier.h>
#include <maya/MDGModifier.h>
#include <maya/MFnIntArrayData.h>
#include <maya/MItMeshPolygon.h>
#include <maya/MItMeshVertex.h>
#include <maya/MDGModifier.h>
#include <ASearchHelper.h>

UntangleCmd::UntangleCmd()
{
	setCommandString("UntangleCmd");
}

UntangleCmd::~UntangleCmd() {}

void* UntangleCmd::creator()
{
	return new UntangleCmd;
}

MSyntax UntangleCmd::newSyntax()
{
	MSyntax syntax;
	return syntax;
}

MStatus UntangleCmd::doIt(const MArgList &args)
{
	MStatus status = parseArgs(args);

	MSelectionList selFaces;
	MGlobal::getActiveSelectionList ( selFaces );
	
	MItSelectionList faceIter( selFaces );
	
	std::vector<int> connections;
	std::map<int, int > vertices;
	int i = 0;
	for ( ; !faceIter.isDone(); faceIter.next() ) {								
        MDagPath item;			
        MObject component;		
        faceIter.getDagPath( item, component );
		addTriangles(connections, vertices, i<<22, 
					item, component);
		i++;
    }
	
	if(connections.size()<1) {
		AHelper::Info<int>("UntangleCmd error zero triangle selected", 0);
		return status;
	}
	
	AHelper::Info<int>("UntangleCmd select n triangles", connections.size() / 3 );
	AHelper::Info<int>("UntangleCmd select n vertices", vertices.size() );
	packVertices(vertices);

/// convert faces to vertices
	MSelectionList selVerts;
	faceIter.reset();
	for ( ; !faceIter.isDone(); faceIter.next() ) {
		MDagPath item;			
        MObject component;		
        faceIter.getDagPath( item, component );
		convertToVert(selVerts, item, component);
	}
		
	MGlobal::setActiveSelectionList ( selVerts, MGlobal::kReplaceList );
	MObject deformer = AHelper::CreateDeformer("untangleDeformer");
	if(deformer.isNull() ) {
		return status;
	}
	
	initDeformer(connections, vertices, deformer);
	
	return status;
}

MStatus UntangleCmd::parseArgs(const MArgList &args)
{
	MStatus status;
	MArgDatabase argData(syntax(), args);
	
	return MStatus::kSuccess;
}

void UntangleCmd::addTriangles(std::vector<int> & connections,
					std::map<int, int > & vertices,
					int vertexIndOffset,
					const MDagPath & mesh, MObject & faces)
{
	MStatus stat;
	MItMeshPolygon iter(mesh, faces, &stat);
	if(!stat) return;
	
	for ( ; !iter.isDone(); iter.next() ) {								
        
		MIntArray indices;
		MPointArray pointArray;
		iter.getTriangles (pointArray, indices);
		for(unsigned i=0; i< indices.length(); ++i) {
			connections.push_back(indices[i]+vertexIndOffset);
			vertices[indices[i]+vertexIndOffset ] = 0;
		}
    }
}

void UntangleCmd::packVertices(std::map<int, int > & vertices)
{
	int i = 0;
	std::map<int, int >::iterator it = vertices.begin();
	for(;it!=vertices.end();++it) {
		it->second = i;
		i++;
	}
}

void UntangleCmd::convertToVert(MSelectionList & selVerts, 
					const MDagPath & mesh, MObject & faces)
{
	MStatus stat;
	MItMeshPolygon iter(mesh, faces, &stat);
	if(!stat) return;
	
	MItMeshVertex vertIter(mesh, MObject::kNullObj, &stat);
	if(!stat) return;
	
	int prevIndex;
	for ( ; !iter.isDone(); iter.next() ) {								
        
		MIntArray indices;
		iter.getVertices (indices);
		for(unsigned i=0; i< indices.length(); ++i) {
			vertIter.setIndex(indices[i], prevIndex);
			MObject vert = vertIter.currentItem();
			selVerts.add (mesh, vert, true);
		}
    }
}

void UntangleCmd::initDeformer(const std::vector<int> & connections,
					const std::map<int, int > & vertices, 
					const MObject & deformer)
{
	MStatus stat;
	MFnDependencyNode fnode(deformer);
	MPlug ntPlug = fnode.findPlug("inNumTriangels");
	ntPlug.setValue((int)connections.size() / 3);
	
	MPlug triPlug = fnode.findPlug("inTriangleIndices");
	
	MFnIntArrayData triD;
	MObject otri = triD.create(vertexIds);
	triPlug.setValue(otri);
	
}
