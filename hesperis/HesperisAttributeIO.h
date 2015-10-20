/*
 *  HesperisAttributeIO.h
 *  opium
 *
 *  Created by jian zhang on 10/18/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include "HesperisAnimIO.h"

class HesperisAttributeIO : public HesperisAnimIO {
public:
	HesperisAttributeIO();
	virtual ~HesperisAttributeIO();
	
	static bool WriteAttributes(const MPlugArray & attribs, HesperisFile * file, const std::string & beheadName = "");
    static bool AddAttribute(const MPlug & attrib, HesperisFile * file, const std::string & beheadName = "");
	
	static bool ReadAttributes(MObject &target = MObject::kNullObj);
	static bool ReadAttributes(HBase * parent, MObject &target);
	
protected:
	
private:
	static bool ReadAttribute(MObject & dst, AAttribute * data, MObject &target);
	static bool ReadStringAttribute(MObject & dst, AStringAttribute * data, MObject &target);
	static bool ReadNumericAttribute(MObject & dst, ANumericAttribute * data, MObject &target);
	static bool ReadCompoundAttribute(MObject & dst, ACompoundAttribute * data, MObject &target);
	static bool ReadEnumAttribute(MObject & dst, AEnumAttribute * data, MObject &target);
};