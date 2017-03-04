/*
 *  ExampleWorks.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 3/1/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "ExampleWorks.h"
#include <AllMama.h>

using namespace aphid;

ProxyViz * ExampleWorks::PtrViz = NULL;
MObject ExampleWorks::ObjViz = MObject::kNullObj;

ExampleWorks::ExampleWorks()
{}

ExampleWorks::~ExampleWorks()
{}

bool ExampleWorks::validateSelection()
{
	MSelectionList slist;
 	MGlobal::getActiveSelectionList( slist );
	if(!validateViz(slist)) {
	    MGlobal::displayWarning("No proxyViz selected");
	}
    if(!PtrViz) {
		return 0;
	}
	
	return 1;
}

bool ExampleWorks::validateViz(const MSelectionList &sels)
{
    MStatus stat;
    MItSelectionList iter(sels, MFn::kPluginLocatorNode, &stat );
    MObject vizobj;
    iter.getDependNode(vizobj);
    if(vizobj != MObject::kNullObj) {
        MFnDependencyNode fviz(vizobj);
		if(fviz.typeName() != "proxyViz") {
			PtrViz = NULL;
			ObjViz = MObject::kNullObj;
			return 0;
		}
		PtrViz = (ProxyViz*)fviz.userNode();
		ObjViz = vizobj;
	}
    
    if(!PtrViz) {
        return 0;
	}
    
    return 1;
}

void ExampleWorks::getConnectExamples(MObjectArray & exmpOs)
{
	MPlug dstPlug;
	AHelper::getNamedPlug(dstPlug, ObjViz, "inExample");
	
	MPlugArray srcPlugs;
	ConnectionHelper::GetArrayPlugInputConnections(srcPlugs, dstPlug);
	
	const int n = srcPlugs.length();
	for(int i=0;i<n;++i) {
		MObject exmpObj = srcPlugs[i].node();
		exmpOs.append(exmpObj);
	}
}

MString ExampleWorks::getExampleStatusStr()
{
	validateSelection();
	if(ObjViz == MObject::kNullObj) {
        return "none"; 
	}
	
	MString res("/.name/default");
	MFnDependencyNode fviz(ObjViz);
	bool isActive = true;
	bool isVisible = true;
	
	AttributeHelper::getBoolAttributeByName(fviz, "exampleActive", isActive);
	addBoolStatusStrSeg(res, isActive, "/.is_active");
	
	AttributeHelper::getBoolAttributeByName(fviz, "exampleVisible", isVisible);
	addBoolStatusStrSeg(res, isVisible, "/.is_visible");
	
	MObjectArray exmpOs;
	getConnectExamples(exmpOs);
	
	const int n = exmpOs.length();
	
	for(int i=0;i<n;++i) {
		MFnDependencyNode fexmp(exmpOs[i]);
		res += "/.name/";
		res += fexmp.name();
		
		AttributeHelper::getBoolAttributeByName(fexmp, "exampleActive", isActive);
		addBoolStatusStrSeg(res, isActive, "/.is_active");
		
		AttributeHelper::getBoolAttributeByName(fexmp, "exampleVisible", isVisible);
		addBoolStatusStrSeg(res, isVisible, "/.is_visible");
		
	}
	
	return res;
}

void ExampleWorks::addBoolStatusStrSeg(MString & res, bool b, const char * segName)
{
	res += segName;
	if(b) {
		res += "/on";
	} else {
		res += "/off";
	}
}

void ExampleWorks::processShowVoxelThreshold(float x)
{
	validateSelection();
	if(ObjViz != MObject::kNullObj) {
		MFnDependencyNode fviz(ObjViz);
		MPlug psho = fviz.findPlug("svt");
		psho.setValue(x);
	}
}

float ExampleWorks::getShowVoxelThreshold()
{
	float r = 1.f;
	validateSelection();
	if(ObjViz != MObject::kNullObj) {
		MFnDependencyNode fviz(ObjViz);
		MPlug psho = fviz.findPlug("svt");
		psho.getValue(r);
	}
	return r;
}
