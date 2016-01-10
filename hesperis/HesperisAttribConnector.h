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

class HesperisAttribConnector {

public:
	static MObject MasterAttribNode;
	
	static void Connect(const std::string & name, ANumericAttribute::NumericAttributeType typ,
								MObject & entity, MObject & attr);
	static void ClearMasterNode();
	
protected:

private:
	static void CreateMasterNode();

};