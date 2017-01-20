/*
 *  ReplacerWorks.cpp
 *  proxyPaint
 *
 *  list objects connected to example 0
 *  each input to shrub is separated by <example>
 *  $viz is long name to proxy viz node
 *  proxyPaintTool -svx 0 -ltr $viz;
 *  begin pick example 0
 *  proxyPaintTool -svx 0 -bpk $viz;
 *  pick example 0 a few times
 *  proxyPaintTool -svx 0 -dpk $viz;
 *  end pick
 *  proxyPaintTool -epk $viz;
 *  get number of picked
 *  proxyPaintTool -gpk $viz;
 *
 *  Created by jian zhang on 1/20/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "ReplacerWorks.h"
#include "proxyVizNode.h"
#include <mama/ConnectionHelper.h>
#include <maya/MFnDagNode.h>
#include <AHelper.h>

using namespace aphid; 

ReplacerWorks::ReplacerWorks()
{}

ReplacerWorks::~ReplacerWorks()
{}

int ReplacerWorks::listInstanceGroup(MStringArray & instanceNames,
					const MObject& node,
					const int & iExample)
{
	if(iExample == 0) {
/// count connections to viz
		return listInstanceTo(instanceNames, node);
	} 
/// count connections to example
	MPlug dstPlug;
	AHelper::getNamedPlug(dstPlug, node, "inExample");
	MPlugArray srcPlugs;
	ConnectionHelper::GetArrayPlugInputConnections(srcPlugs, dstPlug);
	if(srcPlugs.length() < iExample) {
		AHelper::Info<int>("no connection to example", iExample );
		return 0;
	}
	MPlug iexPlug = srcPlugs[iExample - 1];
	MObject iexNode = iexPlug.node();
	MFnDependencyNode fex(iexNode);
	if(fex.typeName() == "shrubViz") {
		return listInstanceToShrub(instanceNames, iexNode);
	} 
	
	return listInstanceTo(instanceNames, iexNode);
}

int ReplacerWorks::countInstanceGroup(ProxyViz * viz,
					const MObject& node,
					const int & iExample)
{
	if(iExample == 0) {
/// count connections to viz
		int ngrp = countInstanceTo(node);
		if(ngrp < 1) {
			return 0;
		}
		viz->clearGroups();
		viz->addGroup(ngrp);
		viz->finishGroups();
		return ngrp;
	} 
/// count connections to example
	MPlug dstPlug;
	AHelper::getNamedPlug(dstPlug, node, "inExample");
	MPlugArray srcPlugs;
	ConnectionHelper::GetArrayPlugInputConnections(srcPlugs, dstPlug);
	if(srcPlugs.length() < iExample) {
		AHelper::Info<int>("no connection to example", iExample );
		return 0;
	}
	MPlug iexPlug = srcPlugs[iExample - 1];
	MObject iexNode = iexPlug.node();
	MFnDependencyNode fex(iexNode);
	if(fex.typeName() == "shrubViz") {
		return countInstanceToShrub(viz, iexNode);
	} 
	
	int ngrp = countInstanceTo(iexNode);
	if(ngrp < 1) {
		return 0;
	}
	viz->clearGroups();
	viz->addGroup(ngrp);
	viz->finishGroups();
	return ngrp;

}

int ReplacerWorks::countInstanceTo(const MObject& node)
{
	MPlug instPlug;
	AHelper::getNamedPlug(instPlug, node, "instanceSpace");
	MPlugArray spacePlugs;
	ConnectionHelper::GetArrayPlugInputConnections(spacePlugs, instPlug);
	int ngrp = spacePlugs.length();
	AHelper::Info<MString>("viz/example", MFnDependencyNode(node).name() );
	AHelper::Info<int>("connected to n object", ngrp );
	return ngrp;	
}

int ReplacerWorks::countInstanceToShrub(ProxyViz * viz,
					const MObject& node)
{
	MPlug dstPlug;
	AHelper::getNamedPlug(dstPlug, node, "inExample");
	MPlugArray srcPlugs;
	ConnectionHelper::GetArrayPlugInputConnections(srcPlugs, dstPlug);
	if(srcPlugs.length() < 1) {
		AHelper::Info<int>("no connection to shrub", 0 );
		return 0;
	}
	
	const int ne = srcPlugs.length();
/// check
	for(int i=0;i<ne;++i) {
		int ngrp = countInstanceTo(srcPlugs[i].node() );
		if(ngrp < 1) {
			AHelper::Info<int>("no connection to input example", i );
			return 0;
		}
	}
	
	int totalNg = 0;
	
	viz->clearGroups();
	
	for(int i=0;i<ne;++i) {
		int ngrp = countInstanceTo(srcPlugs[i].node() );
		viz->addGroup(ngrp);
		totalNg += ngrp;
	
	}
	
	viz->finishGroups();
	AHelper::Info<int>("shrub connected to n object", totalNg );
	return totalNg;
}


int ReplacerWorks::listInstanceTo(MStringArray & instanceNames,
					const MObject& node)
{
	MPlug instPlug;
	AHelper::getNamedPlug(instPlug, node, "instanceSpace");
	MPlugArray spacePlugs;
	ConnectionHelper::GetArrayPlugInputConnections(spacePlugs, instPlug);
	int ngrp = spacePlugs.length();
	for(int i=0;i<ngrp;++i) {
		MObject instObj = spacePlugs[i].node();
		instanceNames.append(MFnDagNode(instObj).partialPathName() );
	}
	return ngrp;	
}

int ReplacerWorks::listInstanceToShrub(MStringArray & instanceNames,
					const MObject& node)
{
	MPlug dstPlug;
	AHelper::getNamedPlug(dstPlug, node, "inExample");
	MPlugArray srcPlugs;
	ConnectionHelper::GetArrayPlugInputConnections(srcPlugs, dstPlug);
	
	const int ne = srcPlugs.length();
	int totalNg = 0;
	for(int i=0;i<ne;++i) {
		instanceNames.append("<example>");
		MObject iexObj = srcPlugs[i].node();
		instanceNames.append(MFnDagNode(iexObj ).partialPathName() );
		int ngrp = listInstanceTo(instanceNames, iexObj );
		totalNg += ngrp;
	
	}
	return totalNg;
}
