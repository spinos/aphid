/*
 *  ShrubWorks.cpp
 *  
 *
 *  Created by jian zhang on 12/26/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "ShrubWorks.h"
#include <maya/MSelectionList.h>
#include <maya/MItMeshVertex.h>
#include <ASearchHelper.h>
#include <gpr/PCASimilarity.h>
#include <gpr/PCAFeature.h>
#include <AllMath.h>

namespace aphid {

ShrubWorks::ShrubWorks()
{}

ShrubWorks::~ShrubWorks()
{}

void ShrubWorks::countMeshNv(int & nv,
					const MDagPath & meshPath) const
{
	MItMeshVertex vertIt(meshPath);
	nv += (int)vertIt.count();
}

void ShrubWorks::getMeshVertices(DenseMatrix<float> * vertices, 
					int iRow, 
					BoundingBox & bbox, 
					const MDagPath & meshPath, 
					const MDagPath & transformPath) const
{
	AHelper::Info<MString>(" get mesh vertices", meshPath.fullPathName() );
	
	MMatrix worldTm = AHelper::GetWorldParentTransformMatrix2(meshPath, transformPath);
	
	Vector3F fpnt;
	MItMeshVertex vertIt(meshPath);
	for(;!vertIt.isDone();vertIt.next() ) {
		
		MPoint pnt = vertIt.position();
		
		pnt *= worldTm;
		
		fpnt.set(pnt.x, pnt.y, pnt.z);
		bbox.expandBy(fpnt);
		
		vertices->copyRow(iRow, (const float *)&fpnt);
		
		iRow++;
	}
	
}

int ShrubWorks::getGroupMeshVertices(DenseMatrix<float> * vertices,
					BoundingBox & bbox, 
					const MDagPath & path) const
{
	AHelper::Info<const char *>(" ShrubWorks get world mesh vertices in group", path.fullPathName().asChar() );
	MDagPathArray meshPaths;
	ASearchHelper::LsAllTypedPaths(meshPaths, path, MFn::kMesh);

	const int n = meshPaths.length();
	if(n < 1) {
		AHelper::Info<MString>(" WARNING ShrubWorks find no mesh in group", path.fullPathName() );
		return 0;
	}
	
	int nv = 0;
	for(int i=0;i<n;++i) {
		countMeshNv(nv, meshPaths[i]);
	}
	
	AHelper::Info<int>(" group mesh n v", nv );
		
	vertices->resize(nv, 3);
	
	int iRow = 0;
	for(int i=0;i<n;++i) {
		getMeshVertices(vertices, iRow, bbox, meshPaths[i], path);
	}
	
	AHelper::Info<BoundingBox>(" group mesh box", bbox );
	
	return nv;
}

void ShrubWorks::addSimilarity(std::vector<SimilarityType * > & similarities, 
					const DenseMatrix<float> & vertices) const
{
	SimilarityType * asim = new SimilarityType();
	similarities.push_back(asim);
	asim->begin(vertices, 2);
}

bool ShrubWorks::findSimilar(std::vector<SimilarityType * > & similarities, 
					const DenseMatrix<float> & vertices) const
{
	const int n = similarities.size();
	if(n<1) {
		return false;
	}
	
	for(int i=0;i<n;++i) {
		if(similarities[i]->select(vertices, 2) ) {
			return true;
		}
	}
	return false;
}

void ShrubWorks::clearSimilarity(std::vector<SimilarityType * > & similarities) const
{
	const int n = similarities.size();
	if(n<1) {
		return;
	}
	
	for(int i=0;i<n;++i) {
		delete similarities[i];
	}
	similarities.clear();
}

void ShrubWorks::separateFeatures(std::vector<SimilarityType * > & similarities) const
{
	const int n = similarities.size();
	if(n<1) {
		return;
	}
	
	for(int i=0;i<n;++i) {
/// K = 2
		similarities[i]->separateFeatures();
	}
}

MStatus ShrubWorks::creatShrub()
{
	MSelectionList selList;
	MGlobal::getActiveSelectionList ( selList );
	if(selList.length() < 1) {
		MGlobal::displayWarning(" ERROR ShrubWorks empty selection, select groups to create shrub");
		return MS::kSuccess;
	}
	
	MDagPathArray paths;
    MItSelectionList iter( selList );
	for ( ; !iter.isDone(); iter.next() ) {								
		MDagPath apath;		
		iter.getDagPath( apath );
        paths.append(apath);
	}
	
	const int ngrp = paths.length();
	AHelper::Info<int>(" ShrubWorks select n groups", ngrp );
	
	std::vector<SimilarityType * > similarities;
	
	BoundingBox totalBox;
	DenseMatrix<float> grpVertD;
	for(int i=0;i<ngrp;++i) {
		BoundingBox grpBox;
		const int nv = getGroupMeshVertices(&grpVertD, grpBox, paths[i]);
		if(nv < 10 ) {
			continue;
		}
		
		if(!findSimilar(similarities, grpVertD)) {
			addSimilarity(similarities, grpVertD);
		}
		
		totalBox.expandBy(grpBox);
	}
	
	const int & ns = similarities.size();
	if(ns < 1) {
		AHelper::Info<int>(" ERROR ShrubWorks found no mesh in selected groups", ns );
		return MS::kSuccess;
	}
	
	AHelper::Info<int>(" ShrubWorks found n similariy", ns );
	
	separateFeatures(similarities);
	
	
	clearSimilarity(similarities);
	return MS::kSuccess;
}

}