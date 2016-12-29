/*
 *  ConnectionHelper.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 12/28/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "ConnectionHelper.h"
#include <AHelper.h>
#include <maya/MDGModifier.h>
#include <maya/MPlugArray.h>

namespace aphid {

ConnectionHelper::ConnectionHelper()
{}

void ConnectionHelper::GetInputConnections(MPlugArray & dst, const MPlug & p)
{
/// as dst not as src
    p.connectedTo ( dst , true, false );
}

void ConnectionHelper::GetOutputConnections(MPlugArray & dst, const MPlug & p)
{ p.connectedTo ( dst , false, true ); }

void ConnectionHelper::GetArrayPlugInputConnections(MPlugArray & dst, const MPlug & p)
{
    if(!p.isArray() ) {
        AHelper::Info<MString>("plug is not array", p.name() );
        return;
    }
    
    unsigned ne = p.numElements();
    for(unsigned i=0;i<ne;++i) {
        MPlugArray ac;
        GetInputConnections(ac, p.elementByPhysicalIndex(i) );
        AHelper::Merge<MPlugArray >(dst, ac);
    }
    
}

void ConnectionHelper::GetAvailablePlug(MPlug & dst, MPlug & p)
{
    const unsigned np = p.evaluateNumElements();
    if(np < 1) {
        dst = p.elementByLogicalIndex(0);
        return;
    }
    
    for(unsigned i=0;i<1000;++i) {
        MPlug ap = p.elementByLogicalIndex(i);
        if(!ap.isConnected() ) {
            AHelper::Info<unsigned>("available elem", i);
            dst = ap;
            return;
        }
    }
}

bool ConnectionHelper::ConnectToArray(MPlug & srcPlug,
							const MObject & dstNode,
							const MString & dstArrayAttrName)
{
	MPlug dstArrayPlug;
	AHelper::getNamedPlug(dstArrayPlug, dstNode, dstArrayAttrName.asChar() );
	if(dstArrayPlug.isNull() ) {
		AHelper::Info<MString>("no destination attrib", dstArrayAttrName);
		return false;
	}
	
	MPlug dstPlug;
	
	if(dstArrayPlug.isArray() ) {
		GetAvailablePlug(dstPlug, dstArrayPlug);
	} else {
		dstPlug = dstArrayPlug;
	}
	
	AHelper::Info<MString>(" connect ", srcPlug.name() );
	AHelper::Info<MString>(" to ", dstPlug.name() );

	MDGModifier modif;
	modif.connect(srcPlug, dstPlug );
	modif.doIt();
	
	return true;
}
	
bool ConnectionHelper::ConnectToArray(const MObject & srcNode,
							const MString & srcAttrName,
							const MObject & dstNode,
							const MString & dstArrayAttrName)
{
	MPlug srcPlug;
	AHelper::getNamedPlug(srcPlug, srcNode, srcAttrName.asChar() );
	if(srcPlug.isNull() ) {
		AHelper::Info<MString>("no source attrib", srcAttrName);
		return false;
	}
	
	return ConnectToArray(srcPlug, dstNode, dstArrayAttrName);
}

bool ConnectionHelper::ConnectedToNode(const MPlug & srcPlug, 
							const MObject & dstNode)
{
	MPlugArray connected;
	GetOutputConnections(connected, srcPlug);
	
	unsigned i = 0;
	for(;i<connected.length();++i) {
		if(connected[i].node() == dstNode) {
			AHelper::Info<MString>(" plug ", srcPlug.name() );
			AHelper::Info<MString>(" connected to", connected[i].name() );
			return true;
		}
	}
	return false;
}

void ConnectionHelper::BreakArrayPlugInputConnections(MPlug & dstPlug)
{
	MPlugArray connected;
	GetArrayPlugInputConnections(connected, dstPlug);
	MDGModifier modif;
	unsigned i = 0;
	for(;i<connected.length();++i) {
		modif.disconnect(connected[i], dstPlug );
		modif.doIt();
	}
}

}