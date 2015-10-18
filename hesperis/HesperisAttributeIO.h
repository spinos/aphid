/*
 *  HesperisAttributeIO.h
 *  opium
 *
 *  Created by jian zhang on 10/18/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include "HesperisIO.h"

class HesperisAttributeIO : public HesperisIO {
public:
	HesperisAttributeIO();
	virtual ~HesperisAttributeIO();
	
	static bool WriteAttributes(const MPlugArray & attribs, HesperisFile * file, const std::string & beheadName = "");
    static bool AddAttribute(const MPlug & attrib, HesperisFile * file, const std::string & beheadName = "");
	
	static bool ReadAttributes(MObject &target = MObject::kNullObj);
protected:
	
private:
	static bool ReadAttributes(HBase * parent, MObject &target);
	static bool ReadAttribute(AAttribute * data, MObject &target);
	static bool ReadStringAttribute(AStringAttribute * data, MObject &target);
	static bool ReadNumericAttribute(ANumericAttribute * data, MObject &target);
	static bool ReadCompoundAttribute(ACompoundAttribute * data, MObject &target);
	static bool ReadEnumAttribute(AEnumAttribute * data, MObject &target);
};