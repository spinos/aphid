/*
 *  AttribUtil.cpp
 *  opium
 *
 *  Created by jian zhang on 10/17/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "AttribUtil.h"
#include <maya/MFnDagNode.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnEnumAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnCompoundAttribute.h>
#include <AHelper.h>
#include <SHelper.h>
#include <HesperisAttributeIO.h>
#include <HesperisAttribConnector.h>
#include <H5IO.h>
#include <AAttributeHelper.h>
#include <boost/tokenizer.hpp>
#include <boost/format.hpp>

AttribNameMap AttribUtil::UserDefinedAttribFilter;

AttribUtil::AttribUtil() {}
AttribUtil::~AttribUtil() {}

void AttribUtil::save3(const MDagPathArray & entities)
{
	unsigned i = 0;
	for(; i< entities.length(); i++) scan(entities[i]);
	if(m_dirtyPlugs.length() < 1) return;
	HesperisAttributeIO::WriteAttributes(m_dirtyPlugs, HesDoc); 
	m_dirtyPlugs.clear();
}

void AttribUtil::scan(const MDagPath &entity)
{		
	MFnDagNode pf(entity);
    AttribNameMap attribs = ListUserDefinedAttribs(pf.fullPathName(), true);
    if(attribs.size() < 1) return;
	
	MPlugArray pls;
    AttribNameMap::const_iterator it = attribs.begin();
    for( ; it != attribs.end(); ++it ) {
        MObject oattrib = pf.attribute(MString(it->first.c_str()));
        MPlug pl(entity.node(), oattrib);
		pls.append(pl);
    }
	AHelper::Merge(m_dirtyPlugs, pls);
}

void AttribUtil::saveUDA(AnimIO& doc)
{
    if( UserDefinedAttribFilter.size() < 1 ) return;
    
    AttribNameMap::const_iterator it = UserDefinedAttribFilter.begin();
    std::stringstream sst;
    for(; it != UserDefinedAttribFilter.end(); ++it) {
        sst<<"|"<<it->first;
    }
    doc.addUDA(sst.str().c_str());
}

void AttribUtil::saveUserDefinedAttrib(AnimIO & doc, MObject & entity)
{
    MFnDagNode pf(entity);
    AttribNameMap attribs = ListUserDefinedAttribs(pf.fullPathName(), true);
    if(attribs.size() < 1) return;
    AttribNameMap::const_iterator it = attribs.begin();
    for( ; it != attribs.end(); ++it ) {
        MObject oattrib = pf.attribute(MString(it->first.c_str()));
        AAttribute::AttributeType t = AAttributeHelper::GetAttribType(oattrib);
        saveAttrib(t, doc, entity, oattrib);
    }
}

void AttribUtil::loadUserDefinedAttrib(AnimIO & doc, MObject & entity)
{
    if(!doc.hasChild())
		return;
	
	MFnDagNode pf(entity);
    AttribNameMap existingAttribs = ListUserDefinedAttribs(pf.fullPathName(), false);
    
	doc.setChildren();
	char hasNext = 1;
	while(hasNext)
	{
		if(doc.checkNodeName("attribute")) 
		    loadAttrib(doc, entity, existingAttribs);
		hasNext = doc.nextNode();
	}
	
	doc.setParent();
}

void AttribUtil::saveAttrib(AAttribute::AttributeType t, AnimIO & doc, const MObject & entity, const MObject & attrib)
{
    switch(t) {
    case AAttribute::aString:
        saveStringAttrib(doc, entity, attrib);
        break;
    case AAttribute::aNumeric:
        saveNumericAttrib(doc, entity, attrib);
        break;
    case AAttribute::aCompound:
        saveCompoundAttrib(doc, entity, attrib);
        break;
    case AAttribute::aEnum:
        saveEnumAttrib(doc, entity, attrib);
        break;
    default:
        MGlobal::displayInfo(MPlug(entity, attrib).name() + " is not supported");
        break;
    }
}

void AttribUtil::saveStringAttrib(AnimIO & doc, const MObject & node, const MObject & attrib)
{
    MFnTypedAttribute fn(attrib);
    if(fn.attrType() != MFnData::kString) {
        MGlobal::displayWarning(fn.name() + " only string typed attrib will be saved");
        return;
    }
    
    MPlug pl(node, attrib);
    MString val;
    pl.getValue(val);
    
    xmlNodePtr p = doc.nodeBegin("attribute");
    doc.setNodeProp(p, "name", fn.name().asChar());
    doc.setNodeProp(p, "short_name", fn.shortName().asChar());
    doc.setNodeProp(p, "attrib_type", "string");
    doc.setNodeProp(p, "value", val.asChar());
    doc.nodeEnd(p);
}

void AttribUtil::saveNumericAttrib(AnimIO & doc, const MObject & node, const MObject & attrib, xmlNodePtr * parent)
{
   MFnNumericAttribute fn(attrib);
   MPlug pl(node, attrib);
   std::string val;
   std::string typ("");
   switch(fn.unitType()) {
       case MFnNumericData::kByte :
           typ = "short";
           val = AAttributeHelper::GetPlugValueAsStr<short>(pl);
           break;
       case MFnNumericData::kShort :
           typ = "short";
           val = AAttributeHelper::GetPlugValueAsStr<short>(pl);
           break;
       case MFnNumericData::kInt:
           typ = "int";
           val = AAttributeHelper::GetPlugValueAsStr<int>(pl);
           break;
       case MFnNumericData::kFloat:
           typ = "float";
           val = AAttributeHelper::GetPlugValueAsStr<float>(pl);
           break;
       case MFnNumericData::kDouble:
           typ = "double";
           val = AAttributeHelper::GetPlugValueAsStr<double>(pl);
           break;
       case MFnNumericData::kBoolean:
           typ = "boolean";
           val = AAttributeHelper::GetPlugValueAsStr<bool>(pl);
           break;
       default:
           MGlobal::displayInfo(fn.name() + " unknown numberic attribute type " + fn.unitType());
           break;
   }
   
   if(typ.size() < 3) return;
   
    xmlNodePtr p = doc.nodeBegin("attribute");
    doc.setNodeProp(p, "name", fn.name().asChar());
    doc.setNodeProp(p, "short_name", fn.shortName().asChar());
    doc.setNodeProp(p, "attrib_type", typ.c_str());
    doc.setNodeProp(p, "value", val.c_str());
    if(parent) doc.nodeEnd(*parent, p);
    else doc.nodeEnd(p);
}

void AttribUtil::saveCompoundAttrib(AnimIO & doc, const MObject & node, const MObject & attrib)
{
    MFnCompoundAttribute fn(attrib);
    xmlNodePtr p = doc.nodeBegin("attribute");
    doc.setNodeProp(p, "name", fn.name().asChar());
    doc.setNodeProp(p, "short_name", fn.shortName().asChar());
    doc.setNodeProp(p, "attrib_type", "compound");
    
    const unsigned n = fn.numChildren();
    unsigned i;
    for(i=0; i < n; i++) {
        MObject child = fn.child(i);
        saveNumericAttrib(doc, node, child, &p);
    }
    doc.nodeEnd(p);
}

void AttribUtil::saveEnumAttrib(AnimIO & doc, const MObject & node, const MObject & attrib)
{
    MFnEnumAttribute fn(attrib);
    short minValue, maxValue;
    fn.getMin(minValue);
 	fn.getMax(maxValue);
 	
 	MPlug pl(node, attrib);
 	std::string val = AAttributeHelper::GetPlugValueAsStr<short>(pl);
 	
 	xmlNodePtr p = doc.nodeBegin("attribute");
    doc.setNodeProp(p, "name", fn.name().asChar());
    doc.setNodeProp(p, "short_name", fn.shortName().asChar());
    doc.setNodeProp(p, "attrib_type", "enum");
    doc.setNodeProp(p, "value", val.c_str());
    
    std::stringstream sst;
    short i;
    for(i = minValue; i <= maxValue; i++) {
        MString fld = fn.fieldName(i);
        xmlNodePtr d = doc.nodeBegin("enum_field");
        doc.setNodeProp(d, "name", fld.asChar());
        sst.str("");
        sst<<i;
        doc.setNodeProp(d, "index", sst.str().c_str());
        doc.nodeEnd(p, d);
    }
    doc.nodeEnd(p);
}

void AttribUtil::loadAttrib(AnimIO & doc, MObject & entity, const AttribNameMap & existingAttribs)
{
    std::string attrName(doc.getAttribByName("name"));
    bool doCreate = false;
    if(existingAttribs.find(attrName) == existingAttribs.end())
        doCreate = true;
    switch( AAttributeHelper::AsAttributeType( doc.getAttribByName("attrib_type") ) ) {
        case AAttribute::aString:
            loadStringAttrib(doc, entity, doCreate);
            break;
        case AAttribute::aEnum:
            loadEnumAttrib(doc, entity, doCreate);
            break;
        case AAttribute::aCompound:
            loadCompoundAttrib(doc, entity, doCreate);
            break;
        case AAttribute::aNumeric:
            loadNumericAttrib(doc, entity, doCreate);
            break;
        default:
            break;
        }
}

void AttribUtil::loadStringAttrib(AnimIO & doc, MObject & entity, bool doCreate)
{
    const MString nameLong(doc.getAttribByName("name"));
    MFnDependencyNode fdep(entity);
    if(doCreate) {
        const MString nameShort(doc.getAttribByName("short_name"));
		MObject attribObject;
		AHelper::createStringAttr(attribObject, nameLong, nameShort);
    }
    MString val(doc.getAttribByName("value"));
    MPlug pl = fdep.findPlug(nameLong);
    pl.setValue(val);
}

void AttribUtil::loadEnumAttrib(AnimIO & doc, MObject & entity, bool doCreate)
{
    if(!doc.hasChild()) return;
    MString nameLong(doc.getAttribByName("name"));
    MFnDependencyNode fdep(entity);
    if(doCreate) {
        MString nameShort(doc.getAttribByName("short_name"));
        MFnEnumAttribute fAttr;
        MObject attribObject = fAttr.create(nameLong, nameShort);
        
        doc.setChildren();
        char hasNext = 1;
        while(hasNext) {
            if(doc.checkNodeName("enum_field")) {
                MString fieldName(doc.getAttribByName("name"));
                short ind = AHelper::GetValueFromStr<short>(doc.getAttribByName("index"));
                fAttr.addField(fieldName, ind);
            }
            hasNext = doc.nextNode();
        }
        doc.setParent();
        
        fAttr.setStorable(true);
        fdep.addAttribute( attribObject, MFnDependencyNode::kLocalDynamicAttr );
    }
    
    MPlug pl = fdep.findPlug(nameLong);
    AAttributeHelper::SetPlugValueFromStr<short>( pl, doc.getAttribByName("value") );
}

void AttribUtil::loadCompoundAttrib(AnimIO & doc, MObject & entity, bool doCreate)
{
    if(!doc.hasChild()) return;
    MString nameLong(doc.getAttribByName("name"));
    MFnDependencyNode fdep(entity);
    char hasNext;
    if(doCreate) {
        MString nameShort(doc.getAttribByName("short_name"));
        MFnCompoundAttribute fAttr;
        MObject attribObject = fAttr.create(nameLong, nameShort);
        
        int numChild = 0;
        doc.setChildren();
        hasNext = 1;
        while(hasNext) {
            if(doc.checkNodeName("attribute")) {
                MObject oattr = createNumericAttrib(doc);
                if(!oattr.isNull()) {
                    fAttr.addChild(oattr);
                    numChild++;
                }
            }
            hasNext = doc.nextNode();
        }
        doc.setParent();
        
        if(numChild < 1) return;
        
        fAttr.setStorable(true);
        fdep.addAttribute( attribObject, MFnDependencyNode::kLocalDynamicAttr );
    }
    
    doc.setChildren();
    hasNext = 1;
    while(hasNext) {
        if(doc.checkNodeName("attribute"))
            setNumericAttrib(fdep, doc);

        hasNext = doc.nextNode();
    }
    doc.setParent();
}

void AttribUtil::loadNumericAttrib(AnimIO & doc, MObject & entity, bool doCreate)
{
    MFnDependencyNode fdep(entity);
    
    if(doCreate) {
        MObject oattr = createNumericAttrib(doc);
        if(oattr.isNull()) return;
    
        fdep.addAttribute( oattr, MFnDependencyNode::kLocalDynamicAttr );
    }
    
    setNumericAttrib(fdep, doc);
}

void AttribUtil::setNumericAttrib(MFnDependencyNode & fdep, AnimIO & doc)
{
    MString nameLong(doc.getAttribByName("name"));
    std::string value(doc.getAttribByName("value"));
    MPlug pl = fdep.findPlug(nameLong);
    
    MFnNumericData::Type typ = AAttributeHelper::GetNumericAttributeType(doc.getAttribByName("attrib_type"));
    switch( typ ) {
        case MFnNumericData::kByte:
            AAttributeHelper::SetPlugValueFromStr<short>( pl, value );
            break;
        case MFnNumericData::kShort:
            AAttributeHelper::SetPlugValueFromStr<short>( pl, value );
            break;
        case MFnNumericData::kInt:
            AAttributeHelper::SetPlugValueFromStr<int>( pl, value );
            break;
        case MFnNumericData::kFloat:
            AAttributeHelper::SetPlugValueFromStr<float>( pl, value );
            break;
        case MFnNumericData::kDouble:
            AAttributeHelper::SetPlugValueFromStr<double>( pl, value );
            break;
        case MFnNumericData::kBoolean:
            AAttributeHelper::SetPlugValueFromStr<bool>( pl, value );
            break;
        default:
            MGlobal::displayWarning(MString("unknown numerica attrib ") + nameLong
                                    + pl.name());
            break;
    }
}

void AttribUtil::ResetUserDefinedAttribFilter(const MString & arg)
{
    UserDefinedAttribFilter.clear();
	if(arg.length() < 1) return;
    typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep("|");
	std::string s(arg.asChar());
	tokenizer tokens(s, sep);
	tokenizer::iterator tok_iter = tokens.begin();
	for (; tok_iter != tokens.end(); ++tok_iter)
        UserDefinedAttribFilter[std::string(*tok_iter)] = 1;
}

AttribNameMap AttribUtil::ListUserDefinedAttribs(const MString & nodeName, bool filtered)
{
    AttribNameMap am;
    if(filtered && IsFilterEmpty()) return am;
    const MString cmd = "listAttr " + nodeName;
    MStringArray res;
    MGlobal::executeCommand(cmd, res);
    unsigned l = res.length();
    unsigned i;
    for(i=0; i < l; i++) {
        std::string k(res[i].asChar());
        // MGlobal::displayInfo(MString("look for ")+res[i]);
        if(filtered) {
            if(UserDefinedAttribFilter.find(k) != UserDefinedAttribFilter.end()) 
                am[k] = 1;
        }
        else 
            am[k] = 1;
    }
    return am;
}

MObject AttribUtil::createNumericAttrib(AnimIO & doc)
{
    MString nameLong(doc.getAttribByName("name"));
    MString nameShort(doc.getAttribByName("short_name"));
    MFnNumericData::Type typ = AAttributeHelper::GetNumericAttributeType(doc.getAttribByName("attrib_type"));
    if(typ == MFnNumericData::kInvalid) {
        MGlobal::displayWarning(nameLong + MString("unknown numberic attrib type ") + doc.getAttribByName("attrib_type"));
        return MObject::kNullObj;
    }
    
    MFnNumericAttribute fAttr;
    MObject attribObject = fAttr.create(nameLong, nameShort, typ);
    fAttr.setStorable(true);
    fAttr.setKeyable(true);
    return attribObject;
}

bool AttribUtil::IsFilterEmpty()
{ return UserDefinedAttribFilter.size() < 1; }

void AttribUtil::saveH5(const std::map<std::string, MDagPath > & entities)
{
	std::map<std::string, MDagPath >::const_iterator ita = entities.begin();
	for(;ita!=entities.end();++ita) scan(ita->second);
	
	if(m_dirtyPlugs.length() < 1) return;
	
	const unsigned n = m_dirtyPlugs.length();
	unsigned i = 0;
	for(;i<n;i++) {
		MObject node = m_dirtyPlugs[i].node();
		H5IO::CreateGroup( HesperisIO::H5PathNameTo(node) );
		saveH5(m_dirtyPlugs[i]);
	}
	
	m_dirtyPlugs.clear();
}

void AttribUtil::saveH5(const MPlug & attrib)
{
	const MObject entity = attrib.node();
	AAttribute::AttributeType t = AAttributeHelper::GetAttribType(attrib.attribute());
	switch(t) {
		case AAttribute::aString:
			saveH5(entity, AAttributeHelper::AsStrData(attrib) );
			break;
		case AAttribute::aNumeric:
			saveH5(entity, AAttributeHelper::AsNumericData(attrib) );
			break;
		case AAttribute::aCompound:
			saveH5(entity, AAttributeHelper::AsCompoundData(attrib) );
			break;
		case AAttribute::aEnum:
			saveH5(entity, AAttributeHelper::AsEnumData(attrib) );
			break;
		default:
			AHelper::Info<std::string >(" attr type not supported ", attrib.name().asChar() );
			break;
    }
	AnimUtil::saveH5(attrib);
}

void AttribUtil::saveH5(const MObject & node, AAttribute * data)
{
	const std::string nodeName = HesperisIO::H5PathNameTo(node);
	if(nodeName.size() < 1) return;
	const std::string attrName = boost::str(boost::format("%1%|%2%") % nodeName % data->shortName() );
	AHelper::Info<std::string>(" w attr ", attrName);
	H5IO::SaveData<HAttributeGroup, AAttribute>(attrName, data);
}

void AttribUtil::load3(const char * filename, MObject & target)
{
	if(!OpenH5(filename, HDocument::oReadAndWrite)) {
		AHelper::Info<const char *>(" uda cannot open h5 file ", filename);
        return;
	}
	
	if(!useH5Bake()) {
        AHelper::Info<const char *>("h5 file is not opened", filename);
        return;   
    }
	
	AHelper::Info<const char *>(" read uda from ", filename);
	HBase w("/");
	HesperisAttributeIO::ReadAttributes(&w, target);
	w.close();
	
	HesperisAttribConnector::ClearMasterNode();
	CloseH5();
}

void AttribUtil::bakeH5(const std::map<std::string, MDagPath > & entities, int flag)
{
	std::map<std::string, MDagPath >::const_iterator ita = entities.begin();
	for(;ita!=entities.end();++ita) scan(ita->second);
	
	if(m_dirtyPlugs.length() < 1) return;
	
	const unsigned n = m_dirtyPlugs.length();
	unsigned i = 0;
	for(;i<n;i++) bakeH5(m_dirtyPlugs[i], flag);
	
	m_dirtyPlugs.clear();
	if(flag == 2) HesperisAttributeIO::ClearBakeData();
}

void AttribUtil::bakeH5(const MPlug & attrib, int flag)
{
	if(IsPlugAnimated (attrib ) ) return;
	MPlugArray conns;
	if(!attrib.connectedTo (conns, true, false )) return;
/// has connection but not key-framed	
	AAttribute::AttributeType t = AAttributeHelper::GetAttribType(attrib.attribute());
	if(t == AAttribute::aNumeric) {
		bakeNumeric(attrib.node(), AAttributeHelper::AsNumericData(attrib), flag );
	}
	else if(t == AAttribute::aEnum) {
		bakeEnum(attrib.node(), AAttributeHelper::AsEnumData(attrib), flag);
	}
	else {
		AHelper::Info<MString >(" attr not bakable ", attrib.name() );
	}
}

void AttribUtil::bakeNumeric(const MObject & entity, ANumericAttribute * data, int flag)
{ 
	const std::string nodeName = HesperisIO::H5PathNameTo(entity);
	const std::string attrName = fullAttrName(nodeName, data);
	if(flag == 0) {
		H5IO::CreateGroup( nodeName );
		HesperisAttributeIO::BeginBakeAttribute(attrName, data);
	}
	else if(flag == 2) {
		HesperisAttributeIO::EndBakeAttribute(attrName, data);
	}
	else {
		HesperisAttributeIO::BakeAttribute(attrName, data);
	}
}

void AttribUtil::bakeEnum(const MObject & entity, AEnumAttribute * data, int flag)
{
	const std::string nodeName = HesperisIO::H5PathNameTo(entity);
	const std::string attrName = fullAttrName(nodeName, data);
	if(flag == 0) {
		H5IO::CreateGroup( nodeName );
		HesperisAttributeIO::BeginBakeEnum(attrName, data);
	}
	else if(flag == 2) {
		HesperisAttributeIO::EndBakeEnum(attrName, data);
	}
	else {
		HesperisAttributeIO::BakeEnum(attrName, data);
	}
}

std::string AttribUtil::fullAttrName(const std::string & nodeName, AAttribute * data) const
{ return boost::str(boost::format("%1%|%2%") % nodeName % data->shortName() ); }

//:~