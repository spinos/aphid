/*
 *  HesperisAttributeIO.cpp
 *  opium
 *
 *  Created by jian zhang on 10/18/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "HesperisAttributeIO.h"
#include <AAttributeHelper.h>
#include <HWorld.h>
#include <HAttributeGroup.h>
#include <boost/format.hpp>

HesperisAttributeIO::HesperisAttributeIO() {}
HesperisAttributeIO::~HesperisAttributeIO() {}

bool HesperisAttributeIO::WriteAttributes(const MPlugArray & attribs, HesperisFile * file, const std::string & beheadName)
{
	file->clearAttributes();
	
	unsigned i = 0;
	for(;i<attribs.length();i++) AddAttribute(attribs[i], file, beheadName);
	
	file->setDirty();
	file->setWriteComponent(HesperisFile::WAttrib);
    bool fstat = file->save();
	if(!fstat) MGlobal::displayWarning(MString(" cannot save attrib to file ")+ file->fileName().c_str());
	file->close();
	return true;
}

bool HesperisAttributeIO::AddAttribute(const MPlug & attrib, HesperisFile * file, const std::string & beheadName)
{
	const std::string nodeName = H5PathNameTo(attrib.node(), beheadName);
	const std::string attrName = boost::str(boost::format("%1%|%2%") % nodeName % attrib.partialName().asChar());
	AHelper::Info<std::string>("att ", attrName);
	
	AAttribute::AttributeType t = AAttributeHelper::GetAttribType(attrib.attribute());
	switch(t) {
		case AAttribute::aString:
			file->addAttribute(attrName, AAttributeHelper::AsStrData(attrib));
			break;
		case AAttribute::aNumeric:
			file->addAttribute(attrName, AAttributeHelper::AsNumericData(attrib));
			break;
		case AAttribute::aCompound:
			file->addAttribute(attrName, AAttributeHelper::AsCompoundData(attrib));
			break;
		case AAttribute::aEnum:
			file->addAttribute(attrName, AAttributeHelper::AsEnumData(attrib));
			break;
		default:
			AHelper::Info<std::string>("attr type not supported ", attrName);
			break;
    }

	return true;
}

bool HesperisAttributeIO::ReadAttributes(MObject &target)
{
	MGlobal::displayInfo("opium v3 read attribute");
    HWorld grpWorld;
    ReadAttributes(&grpWorld, target);
    grpWorld.close();
    return true;
}

bool HesperisAttributeIO::ReadAttributes(HBase * parent, MObject &target)
{
	std::vector<std::string > allGrps;
	std::vector<std::string > allAttrs;
	parent->lsTypedChild<HBase>(allGrps);
	std::vector<std::string>::const_iterator it = allGrps.begin();
	for(;it!=allGrps.end();++it) {
		HBase child(*it);
		if(child.hasNamedAttr(".attr_typ")) {
			allAttrs.push_back(*it);
		}
		else {
			MObject otm;
			if( FindNamedChild(otm, child.lastName(), target) )
				ReadAttributes(&child, otm);
		}	
		child.close();
	}
	
	it = allAttrs.begin();
	for(;it!=allAttrs.end();++it) {
		HAttributeGroup a(*it);
		AAttributeWrap wrap;
		if(a.load(wrap)) {
			MObject attr;
			if(ReadAttribute(attr, wrap.attrib(), target))
				ReadAnimation(&a, target, attr);
		}
		a.close();
	}
	
	allAttrs.clear();
	allGrps.clear();
	
	return true;
}

bool HesperisAttributeIO::ReadAttribute(MObject & dst, AAttribute * data, MObject &target)
{
	switch(data->attrType()) {
		case AAttribute::aString:
			ReadStringAttribute(dst, static_cast<AStringAttribute *> (data), target);
			break;
		case AAttribute::aNumeric:
			ReadNumericAttribute(dst, static_cast<ANumericAttribute *> (data), target);
			break;
		case AAttribute::aCompound:
			ReadCompoundAttribute(dst, static_cast<ACompoundAttribute *> (data), target);
			break;
		case AAttribute::aEnum:
			ReadEnumAttribute(dst, static_cast<AEnumAttribute *> (data), target);
			break;
		default:
			break;
    }
	return true;
}

bool HesperisAttributeIO::ReadStringAttribute(MObject & dst, AStringAttribute * data, MObject &target)
{
	if(AAttributeHelper::HasNamedAttribute(dst, target, data->shortName() )) {
		if(!AAttributeHelper::IsStringAttr(dst) ) {
			AHelper::Info<std::string >(" existing attrib is not string ", data->longName() );
			return false;
		}
	}
	else {
		if(!AAttributeHelper::AddStringAttr(dst, target,
							data->longName(),
							data->shortName())) {
			AHelper::Info<std::string >(" cannot create string attrib ", data->longName() );
			return false;
		}
	}
	
	MPlug(target, dst).setValue(data->value().c_str() );
	return true;
}

bool HesperisAttributeIO::ReadNumericAttribute(MObject & dst, ANumericAttribute * data, MObject &target)
{
	MFnNumericData::Type dt = MFnNumericData::kInvalid;
	switch(data->numericType()) {
		case ANumericAttribute::TByteNumeric:
			dt = MFnNumericData::kByte;
			break;
		case ANumericAttribute::TShortNumeric:
			dt = MFnNumericData::kShort;
			break;
		case ANumericAttribute::TIntNumeric:
			dt = MFnNumericData::kInt;
			break;
		case ANumericAttribute::TBooleanNumeric:
			dt = MFnNumericData::kBoolean;
			break;
		case ANumericAttribute::TFloatNumeric:
			dt = MFnNumericData::kFloat;
			break;
		case ANumericAttribute::TDoubleNumeric:
			dt = MFnNumericData::kDouble;
			break;
		default:
			break;
    }
	if(dt == MFnNumericData::kInvalid) return false;

	if(AAttributeHelper::HasNamedAttribute(dst, target, data->shortName() )) {
		if(!AAttributeHelper::IsNumericAttr(dst, dt) ) {
			AHelper::Info<std::string >(" existing attrib is not correct numeric type ", data->longName() );
			return false;
		}
	}
	else {
		if(!AAttributeHelper::AddNumericAttr(dst, target,
							data->longName(),
							data->shortName(),
							dt) ) {
			AHelper::Info<std::string >(" cannot create numeric attrib ", data->longName() );
			return false;
		}
	}
	
	short va;
	int vb;
	float vc;
	double vd;
	bool ve;
	MPlug pg(target, dst);
	switch(data->numericType()) {
		case ANumericAttribute::TByteNumeric:
			va = (static_cast<AByteNumericAttribute *> (data))->value();
			pg.setValue(va);
			break;
		case ANumericAttribute::TShortNumeric:
			va = (static_cast<AShortNumericAttribute *> (data))->value();
			pg.setValue(va);
			break;
		case ANumericAttribute::TIntNumeric:
			vb = (static_cast<AIntNumericAttribute *> (data))->value();
			pg.setValue(vb);
			break;
		case ANumericAttribute::TBooleanNumeric:
			ve = (static_cast<ABooleanNumericAttribute *> (data))->value();
			pg.setValue(ve);
			break;
		case ANumericAttribute::TFloatNumeric:
			vc = (static_cast<AFloatNumericAttribute *> (data))->value();
			pg.setValue(vc);
			break;
		case ANumericAttribute::TDoubleNumeric:
			vd = (static_cast<ADoubleNumericAttribute *> (data))->value();
			pg.setValue(vd);
			break;
		default:
			break;
    }
	
	return true;
}

bool HesperisAttributeIO::ReadCompoundAttribute(MObject & dst, ACompoundAttribute * data, MObject &target)
{
	AHelper::Info<std::string >(" todo compound attrib ", data->longName() );
	return false;
}

bool HesperisAttributeIO::ReadEnumAttribute(MObject & dst, AEnumAttribute * data, MObject &target)
{
	if(data->numFields() < 1) {
		AHelper::Info<std::string >(" enum attrib has no field ", data->longName() );
		return false;
	}
	
	short a, b, i;
	short v = data->value(a, b);
		
	if(AAttributeHelper::HasNamedAttribute(dst, target, data->shortName() )) {
		if(!AAttributeHelper::IsEnumAttr(dst) ) {
			AHelper::Info<std::string >(" existing attrib is not enum ", data->longName() );
			return false;
		}
	}
	else {
		std::map<short, std::string > fld;
		for(i=a; i<=b; i++) {
			fld[i] = data->fieldName(i);
		}
		
		if(!AAttributeHelper::AddEnumAttr(dst, target,
							data->longName(),
							data->shortName(),
							fld )) {
			AHelper::Info<std::string >(" cannot create enum attrib ", data->longName() );
			return false;
		}
		fld.clear();
	}
	
	MPlug(target, dst).setValue(v);
	return true;
}
//:~