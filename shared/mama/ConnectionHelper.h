/*
 *  ConnectionHelper.h
 *  proxyPaint
 *
 *  Created by jian zhang on 12/28/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_MAMA_CONNECTION_HELPER_H
#define APH_MAMA_CONNECTION_HELPER_H

class MObject;
class MString;
class MPlug;
class MPlugArray;

namespace aphid {

class ConnectionHelper {

public:
	ConnectionHelper();
	
	static void GetInputConnections(MPlugArray & dst, const MPlug & p);
	static void GetOutputConnections(MPlugArray & dst, const MPlug & p);
	static void GetArrayPlugInputConnections(MPlugArray & dst, const MPlug & p);
/// element of p as well
	static void GetArrayPlugInputConnections2(MPlugArray & dst, 
	                                        MPlugArray & elm, const MPlug & p);
	
/// first in array not connected
	static void GetAvailablePlug(MPlug & dst, MPlug & p);
/// to prefered physical index slot
	static bool ConnectToArray(MPlug & srcPlug,
							MPlug & dstArrayPlug,
							const int & refSlot = -1);
	
	static bool ConnectToArray(MPlug & srcPlug,
							const MObject & dstNode,
							const MString & dstArrayAttrName,
							const int & refSlot = -1);
				
	static bool ConnectToArray(const MObject & srcNode,
							const MString & srcAttrName,
							const MObject & dstNode,
							const MString & dstArrayAttrName,
							const int & refSlot = -1);
/// write logical index of source plug connected to destination plug if necessary
	static bool ConnectedToNode(const MPlug & srcPlug, 
							const MObject & dstNode,
							int * outSlot = 0);
							
	static void BreakArrayPlugInputConnections(MPlug & dstPlug);
	
	static bool ConnnectArrayOneToOne(MPlugArray & srcPlugs, 
	                        const MObject & dstNode,
							const MString & dstArrayAttrName);
/// if specified attr is array, test referred slot							
	static MObject GetConnectedNode(const MObject & node,
							const MString & attrName,
							const int & refSlot = -1);
							
	static void BreakInputConnection(MPlug & dstPlug);
/// to prefered (sparse) logical index slot or break all by default
	static void BreakArrayPlugInputConnections(const MObject & dstNode,
							const MString & dstArrayAttrName,
							const int & refSlot = -1);
							
};

}

#endif