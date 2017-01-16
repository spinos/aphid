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

void ConnectionHelper::GetArrayPlugInputConnections2(MPlugArray & dst, 
	                                        MPlugArray & elm, const MPlug & p)
{
    if(!p.isArray() ) {
        AHelper::Info<MString>("plug is not array", p.name() );
        return;
    }
    dst.clear();
    unsigned ne = p.numElements();
    for(unsigned i=0;i<ne;++i) {
        MPlugArray ac;
        GetInputConnections(ac, p.elementByPhysicalIndex(i) );
        if(ac.length() > 0) {
            dst.append(ac[0]);
            elm.append(p.elementByPhysicalIndex(i) );
        }
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
							MPlug & dstArrayPlug,
							const int & refSlot)
{
    MPlug dstPlug;
	MStatus stat;
	if(dstArrayPlug.isArray() ) {
		if(refSlot < 0) {
			GetAvailablePlug(dstPlug, dstArrayPlug);
		} else {
			dstPlug = dstArrayPlug.elementByPhysicalIndex((unsigned)refSlot, &stat);
			if(!stat) {
				AHelper::Info<MString>(" ERROR destinate plug ", dstArrayPlug.name() );
				AHelper::Info<int>(" has no slot ", refSlot );
				return false;
			}
		}
	} else {
		dstPlug = dstArrayPlug;
	}
	
	MDGModifier modif;
	modif.connect(srcPlug, dstPlug );
	stat = modif.doIt();
	if(!stat) {
	    AHelper::Info<MString>(" ERROR cannot connect ", srcPlug.name() );   
		AHelper::Info<MString>(" to ", dstPlug.name() );  
		return false; 
	}
	
	AHelper::Info<MString>(" connect ", srcPlug.name() );
	AHelper::Info<MString>(" to ", dstPlug.name() );

	return true;
}

bool ConnectionHelper::ConnectToArray(MPlug & srcPlug,
							const MObject & dstNode,
							const MString & dstArrayAttrName,
							const int & refSlot)
{
	MPlug dstArrayPlug;
	AHelper::getNamedPlug(dstArrayPlug, dstNode, dstArrayAttrName.asChar() );
	if(dstArrayPlug.isNull() ) {
		AHelper::Info<MString>(" destination node", MFnDependencyNode(dstNode).name() );
		AHelper::Info<MString>(" has no attrib", dstArrayAttrName);
		return false;
	}

	return ConnectToArray(srcPlug, dstArrayPlug, refSlot);
}
	
bool ConnectionHelper::ConnectToArray(const MObject & srcNode,
							const MString & srcAttrName,
							const MObject & dstNode,
							const MString & dstArrayAttrName,
							const int & refSlot)
{
	MPlug srcPlug;
	AHelper::getNamedPlug(srcPlug, srcNode, srcAttrName.asChar() );
	if(srcPlug.isNull() ) {
		AHelper::Info<MString>(" source node", MFnDependencyNode(srcNode).name() );
		AHelper::Info<MString>(" has no attrib", srcAttrName);
		return false;
	}
	
	return ConnectToArray(srcPlug, dstNode, dstArrayAttrName, refSlot);
}

bool ConnectionHelper::ConnectedToNode(const MPlug & srcPlug, 
							const MObject & dstNode,
							int * outSlot)
{
	MPlugArray connected;
	GetOutputConnections(connected, srcPlug);
	
	unsigned i = 0;
	for(;i<connected.length();++i) {
		if(connected[i].node() == dstNode) {
			AHelper::Info<MString>(" plug ", srcPlug.name() );
			AHelper::Info<MString>(" is connected to", connected[i].name() );
			if(outSlot && connected[i].isElement() ) {
				unsigned phyInd = connected[i].logicalIndex();
				*outSlot = (int)phyInd;
				AHelper::Info<int>(" physical index ", *outSlot );

			}
			return true;
		}
	}
	return false;
}

void ConnectionHelper::BreakArrayPlugInputConnections(MPlug & dstPlug)
{
	MPlugArray connectedFrom;
	MPlugArray connectedTo;
	GetArrayPlugInputConnections2(connectedFrom, connectedTo, dstPlug);
	const int n = connectedFrom.length();
	if(n<1) {
	    return;    
	}
	
	AHelper::Info<int>(" break n connection", n );
	AHelper::Info<MString>(" to", dstPlug.name() );
		     
	MDGModifier modif;

	for(int i=0;i<n;++i) {
	    AHelper::Info<MString>(" disconnect", connectedFrom[i].name() );
		AHelper::Info<MString>(" and", connectedTo[i].name() );
		     
		modif.disconnect(connectedFrom[i], connectedTo[i] );
		
		if(!modif.doIt() ) {
		     AHelper::Info<MString>(" WARNING cannot disconnect", connectedFrom[i].name() );
		     AHelper::Info<MString>(" and", connectedTo[i].name() );
		}
	}
}

bool ConnectionHelper::ConnnectArrayOneToOne(MPlugArray & srcPlugs, 
	                        const MObject & dstNode,
							const MString & dstArrayAttrName)
{
    MPlug dstArrPlug;
	AHelper::getNamedPlug(dstArrPlug, dstNode, dstArrayAttrName.asChar() );
	if(dstArrPlug.isNull() ) {
	    AHelper::Info<MString>("no destination attrib", dstArrayAttrName);
		return false;
	}
	
	BreakArrayPlugInputConnections(dstArrPlug);
	
	const int n = srcPlugs.length();
	for(int i=0; i<n;++i) {
		MPlug src = srcPlugs[i];
	    ConnectToArray(src, dstArrPlug);
	}
	
	return true;
}

MObject ConnectionHelper::GetConnectedNode(const MObject & node,
							const MString & attrName,
							const int & refSlot)
{
	MPlug attrPlug;
	AHelper::getNamedPlug(attrPlug, node, attrName.asChar() );
	if(attrPlug.isNull() ) {
	    AHelper::Info<MString>("no destination attrib", attrName);
		return MObject::kNullObj;
	}
	
	MPlugArray srcPlugs;
	if(attrPlug.isArray() ) {
		GetArrayPlugInputConnections(srcPlugs, attrPlug );
	} else {
		GetInputConnections(srcPlugs, attrPlug );
	}
	
	if(srcPlugs.length() < 1) {
		return MObject::kNullObj;
	}
	
	if(!attrPlug.isArray() || refSlot < 0) {
		return srcPlugs[0].node();
	}
	
	int j = srcPlugs.length() - 1;
	if(refSlot > j) {
		AHelper::Info<MString>("destination attrib out of referred slot", attrName);
		AHelper::Info<int>("last", j);
		return MObject::kNullObj;
	}
	
	return srcPlugs[refSlot].node();
	
}

}