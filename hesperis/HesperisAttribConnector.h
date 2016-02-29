/*
 *  HesperisAttribConnector.h
 *  opium
 *
 *  Created by jian zhang on 1/10/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <maya/MObject.h>
#include <AAttribute.h>

namespace aphid {
    
class HesperisAttribConnector {

public:
	static MObject MasterAttribNode;
	
	static void ConnectNumeric(const std::string & name, ANumericAttribute::NumericAttributeType typ,
								MObject & entity, MObject & attr);
	static void ConnectEnum(const std::string & name, MObject & entity, MObject & attr);
	static void ClearMasterNode();
	
protected:

private:
	static void CreateMasterNode();
	static void Connect(const std::string & name, const std::string & plgName, const std::string & srcName,
							MObject & entity, MObject & attr);
};

}