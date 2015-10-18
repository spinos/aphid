/*
 *  HAttributeGroup.cpp
 *  aphid
 *
 *  Created by jian zhang on 10/18/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "HAttributeGroup.h"
#include <sstream>

HAttributeGroup::HAttributeGroup(const std::string & path) : HBase(path) {}
HAttributeGroup::~HAttributeGroup() {}

char HAttributeGroup::verifyType()
{ 
	if(!hasNamedAttr(".attr_typ"))
		return 0;
	return 1; 
}

char HAttributeGroup::save(AAttribute * data)
{
	if(!hasNamedAttr(".attr_typ"))
		addIntAttr(".attr_typ");
		
	int t = data->attrType();
	writeIntAttr(".attr_typ", &t);
	
	std::cout<<"\nsave attr type "<<data->attrTypeStr();
	
	if(!hasNamedAttr(".longname"))
		addStringAttr(".longname", data->longName().size());
	writeStringAttr(".longname", data->longName());
		
	std::cout<<"\nsave attr name "<<data->longName();
	
	if(data->isNumeric()) writeNumeric( static_cast<ANumericAttribute *> (data) );
	else if(data->isEnum()) writeEnum( static_cast<AEnumAttribute *> (data) );
	else if(data->isString()) writeString( static_cast<AStringAttribute *> (data) );
	else if(data->isCompound()) writeCompound( static_cast<ACompoundAttribute *> (data) );
	return 1; 
}

void HAttributeGroup::writeNumeric(ANumericAttribute * data)
{
	int t = data->numericType();
	if(t==0) return;

	if(!hasNamedAttr(".num_typ"))
		addIntAttr(".num_typ");
	
	writeIntAttr(".num_typ", &t);
	
	switch (data->numericType()) {
		case ANumericAttribute::TShortNumeric:
		case ANumericAttribute::TIntNumeric:
		case ANumericAttribute::TBooleanNumeric:
			writeNumericValueAsInt(data);
			break;
		case ANumericAttribute::TFloatNumeric:
		case ANumericAttribute::TDoubleNumeric:
			writeNumericValueAsFlt(data);
			break;
		default:
			break;
	}
}

void HAttributeGroup::writeNumericValueAsInt(ANumericAttribute * data)
{
	short va;
	int vb;
	bool vc;
	switch (data->numericType()) {
		case ANumericAttribute::TShortNumeric:
			va = (static_cast<AShortNumericAttribute *> (data))->value();
			vb = va;
			break;
		case ANumericAttribute::TIntNumeric:
			vb = (static_cast<AIntNumericAttribute *> (data))->value();
			break;
		case ANumericAttribute::TBooleanNumeric:
			vc = (static_cast<ABooleanNumericAttribute *> (data))->value();
			vb = vc;
			break;
		default:
			break;
	}
	if(!hasNamedAttr(".val"))
		addIntAttr(".val");
	
	writeIntAttr(".val", &vb);
	std::cout<<" value "<<vb;
}

void HAttributeGroup::writeNumericValueAsFlt(ANumericAttribute * data)
{
	float va;
	double vb;
	switch (data->numericType()) {
		case ANumericAttribute::TFloatNumeric:
			va = (static_cast<AFloatNumericAttribute *> (data))->value();
			break;
		case ANumericAttribute::TDoubleNumeric:
			vb = (static_cast<ADoubleNumericAttribute *> (data))->value();
			va = vb;
			break;
		default:
			break;
	}
	if(!hasNamedAttr(".val"))
		addFloatAttr(".val");
	
	writeFloatAttr(".val", &va);
	std::cout<<" value "<<va;
}

void HAttributeGroup::writeEnum(AEnumAttribute * data)
{
	short a, b;
	int v = data->value(a, b);
	if(!hasNamedAttr(".val"))
		addIntAttr(".val");
	writeIntAttr(".val", &v);
	
	int r[2];
	r[0] = a;
	r[1] = b;
	
	if(!hasNamedAttr(".range"))
		addIntAttr(".range", 2);
	writeIntAttr(".range", r);	
	
	std::cout<<" value "<<v;
	std::cout<<" range "<<a<<":"<<b;
	
	short i;
	std::stringstream sst;
	for(i=a; i<=b; i++) {
		std::string fn = data->fieldName(i);
		sst.str("");
		sst<<i;
		if(!hasNamedAttr(sst.str().c_str()))
			addStringAttr(sst.str().c_str(), fn.size());
		writeStringAttr(sst.str().c_str(), fn);
		std::cout<<" field "<<i<<":"<<fn;
	}
}

void HAttributeGroup::writeString(AStringAttribute * data)
{
	std::string v = data->value();
	if(!hasNamedAttr(".val"))
		addStringAttr(".val", v.size());
	writeStringAttr(".val", v);
	std::cout<<" value "<<v;
}

void HAttributeGroup::writeCompound(ACompoundAttribute * data)
{
	short n = data->numChild();
	short i=0;
	for(;i<n;i++) {
		HAttributeGroup g(childPath(data->shortName()));
		g.save(data->child(i));
		g.close();
	}
}

char HAttributeGroup::load(AAttributeWrap & wrap)
{
	int t = 0;
	readIntAttr(".attr_typ", &t);
	
	if(t == AAttribute::aNumeric) {
		int tn = 0;
		if(hasNamedAttr(".num_typ")) {
			readIntAttr(".num_typ", &tn);
			loadNumeric( wrap.createNumeric(tn) );
		}
	}
	else if(t == AAttribute::aEnum) {
		loadEnum( wrap.createEnum() );
	}
	else if(t == AAttribute::aString) {
		loadString( wrap.createString() );
	}
	else if(t == AAttribute::aCompound) {
		loadCompound( wrap.createCompound() );
	}
	
	AAttribute * data = wrap.attrib();
	if(!data) return false;
	
	std::string lnm;
	readStringAttr(".longname", lnm);
	
	data->setLongName(lnm);
	data->setShortName(lastName());
	return 1; 
}

bool HAttributeGroup::loadNumeric(ANumericAttribute * data)
{
	switch (data->numericType()) {
		case ANumericAttribute::TShortNumeric:
		case ANumericAttribute::TIntNumeric:
		case ANumericAttribute::TBooleanNumeric:
			readNumericValueAsInt(data);
			break;
		case ANumericAttribute::TFloatNumeric:
		case ANumericAttribute::TDoubleNumeric:
			readNumericValueAsFlt(data);
			break;
		default:
			break;
	}
	return true;
}

bool HAttributeGroup::loadEnum(AEnumAttribute * data)
{
	if(!hasNamedAttr(".val")) return false;
	int v = 0;
	readIntAttr(".val", &v);
	
	data->setValue(v);
	
	int r[2];
	r[0] = 0;
	r[1] = 1;
	
	readIntAttr(".range", r);
	
	short a = r[0];
	short b = r[1];
	data->setRange(a, b);
	
	short i;
	std::stringstream sst;
	for(i=a; i<=b; i++) {
		sst.str("");
		sst<<i;
		std::string fn;
		if(hasNamedAttr(sst.str().c_str())) {
			readStringAttr(sst.str().c_str(), fn);
			data->addField(i, fn);
			std::cout<<" field "<<i<<":"<<fn;
		}
	}
	return true;
}

bool HAttributeGroup::loadString(AStringAttribute * data)
{
	if(!hasNamedAttr(".val")) return false;
		
	std::string v;
	readStringAttr(".val", v);
	data->setValue(v);
	std::cout<<" value "<<v;
	return true;
}

bool HAttributeGroup::loadCompound(ACompoundAttribute * data)
{
	std::vector<std::string > names;
	lsTypedChild<HAttributeGroup>(names);
	if(names.size() < 1) return false;
	
	std::vector<std::string >::iterator it = names.begin();
	for(; it != names.end(); ++it) {
		HAttributeGroup g(*it);
		AAttributeWrap wrap;
		g.load(wrap);
		g.close();
		
		data->addChild(wrap.attrib());
	}
	return true;
}

bool HAttributeGroup::readNumericValueAsInt(ANumericAttribute * data)
{
	if(!hasNamedAttr(".val")) return false;
	int v = 0;
	readIntAttr(".val", &v);
	data->setValue(v);
	return true;
}

bool HAttributeGroup::readNumericValueAsFlt(ANumericAttribute * data)
{
	if(!hasNamedAttr(".val")) return false;
	float v;
	readFloatAttr(".val", &v);
	data->setValue(v);
	return true;
}

AAttributeWrap::AAttributeWrap() 
{ m_attrib = NULL; }

AAttributeWrap::~AAttributeWrap() {}

AStringAttribute * AAttributeWrap::createString()
{ 
	AStringAttribute * r = new AStringAttribute;
	m_attrib = r;
	return r; 
}

AEnumAttribute * AAttributeWrap::createEnum()
{
	AEnumAttribute * r = new AEnumAttribute;
	m_attrib = r;
	return r;
}

ACompoundAttribute * AAttributeWrap::createCompound()
{
	ACompoundAttribute * r = new ACompoundAttribute;
	m_attrib = r;
	return r;
}

ANumericAttribute * AAttributeWrap::createNumeric(int numericType)
{
	ANumericAttribute * r = NULL;
	if( numericType == ANumericAttribute::TShortNumeric ) {
		r = new AShortNumericAttribute;
	}
	else if( numericType == ANumericAttribute::TIntNumeric ) {
		r = new AIntNumericAttribute;
	}
	else if( numericType == ANumericAttribute::TFloatNumeric ) {
		r = new AFloatNumericAttribute;
	}
	else if( numericType == ANumericAttribute::TDoubleNumeric ) {
		r = new ADoubleNumericAttribute;
	}
	else if( numericType == ANumericAttribute::TBooleanNumeric ) {
		r = new ABooleanNumericAttribute;
	}

	m_attrib = r;
	return r;
}

AAttribute * AAttributeWrap::attrib()
{ return m_attrib; }
//:~