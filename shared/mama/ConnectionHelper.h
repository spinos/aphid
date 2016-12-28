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
	
/// first in array not connected
	static void GetAvailablePlug(MPlug & dst, MPlug & p);
	
	static bool ConnectToArray(const MObject & srcNode,
							const MString & srcAttrName,
							const MObject & dstNode,
							const MString & dstArrayAttrName);
	
};

}

#endif