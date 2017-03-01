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

MString ExampleWorks::getExampleStatusStr()
{
	validateSelection();
	if(PtrViz) {
        return "todo";
	}
	return "none";
}
