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
#include <HNumericBundle.h>
#include <HOocArray.h>
#include <boost/format.hpp>
#include "HesperisAttribConnector.h"

namespace aphid {
    
std::map<std::string, HObject * > HesperisAttributeIO::MappedBakeData;

HesperisAttributeIO::HesperisAttributeIO() {}
HesperisAttributeIO::~HesperisAttributeIO() {}

void buildBundle(std::vector<ANumericAttribute * > & bundle,
                            const std::string & attrName, 
                            const MPlugArray & attribs)
{
    const int n = attribs.length();
    for(int i=0;i<n;++i) {
        const MPlug & attrib = attribs[i];
        if(attrName != std::string(attribs[i].partialName().asChar() ) ) 
            continue;
            
        ANumericAttribute * data = NULL;
        AAttribute::AttributeType t = AAttributeHelper::GetAttribType(attrib.attribute());
        if(t == AAttribute::aNumeric)
		    data = AAttributeHelper::AsNumericData(attrib);
		else
		    AHelper::Info<std::string>("\n HesperisAttributeIO build bundle attr type not supported", attrName);
		       
        if(data)
            bundle.push_back(data);    
        
    }
}

void saveBundle(const std::vector<ANumericAttribute * > & bd,
                const std::string & name,
                const std::string & parentName,
                HesperisFile * file)
{  
    const int n = bd.size();
    ANumericAttribute::NumericAttributeType nt = bd[0]->numericType();
    ABundleAttribute bnd;
    bnd.create(n, nt);
    
    if(bnd.numericType() == ANumericAttribute::TUnkownNumeric) {
        AHelper::Info<std::string>("\n HesperisAttributeIO numeric attr type not supported", name);
        return;
    }
        
    bnd.setLongName(bd[0]->longName() );
    bnd.setShortName(bd[0]->shortName() );
    
    int i=0;
    for(;i<n;++i) {
        
        if(nt == ANumericAttribute::TDoubleNumeric) {
            double dv = ((ADoubleNumericAttribute *)bd[i])->value();
            memcpy(&bnd.value()[i], &dv, sizeof(double) );
        }
        else if(nt == ANumericAttribute::TBooleanNumeric) {
            bool bv = ((ABooleanNumericAttribute *)bd[i])->value();
            memcpy(&bnd.value()[i], &bv, sizeof(bool) );
        }
        else if(nt == ANumericAttribute::TFloatNumeric) {
            float fv = ((AFloatNumericAttribute *)bd[i])->value();
            memcpy(&bnd.value()[i], &fv, sizeof(float) );
        }
        else if(nt == ANumericAttribute::TIntNumeric) {
            int iv = ((AIntNumericAttribute *)bd[i])->value();
            memcpy(&bnd.value()[i], &iv, sizeof(int) );
        }
        else if(nt == ANumericAttribute::TShortNumeric) {
            short sv = ((AShortNumericAttribute *)bd[i])->value();
            memcpy(&bnd.value()[i], &sv, sizeof(short) );
        }
        else if(nt == ANumericAttribute::TByteNumeric) {
            char cv = ((AByteNumericAttribute *)bd[i])->asChar();
            memcpy(&bnd.value()[i], &cv, sizeof(char) );
        }
        
    }

    AHelper::Info<std::string>("\n HesperisAttributeIO save bundle", name);

    file->setBundleEntry(&bnd, parentName);
	file->setDirty();
	file->setWriteComponent(HesperisFile::WBundle);
    bool fstat = file->save();
	if(!fstat) 
	    AHelper::Info<std::string>("\n HesperisAttributeIO cannot save attrib bundle to file ", file->fileName() );

}

void clearBundle(std::vector<ANumericAttribute * > & bundle)
{
    std::vector<ANumericAttribute * >::iterator it = bundle.begin();
    for(;it!=bundle.end();++it)
        delete *it;
        
    bundle.clear(); 
}

bool HesperisAttributeIO::WriteAttributeBoundle(const std::map<std::string, char> & attrNames,
                            const MPlugArray & attribs,
                            const std::string & parentName,
                            HesperisFile * file)
{
    std::cout<<"HesperisAttributeIO write bundle n entity "<<attribs.length()
            <<" n filter "<<attrNames.size();
            
    std::vector<ANumericAttribute * > bd;
/// assuming every entity has named attribs
    std::map<std::string, char>::const_iterator it = attrNames.begin();
    for(;it!=attrNames.end();++it) {
        buildBundle(bd, it->first, attribs);
        if(bd.size() > 0) {
            saveBundle(bd, it->first, parentName, file);
            clearBundle(bd);
        }
    }
    std::cout.flush();
    return true;
}

bool HesperisAttributeIO::WriteAttributes(const MPlugArray & attribs, HesperisFile * file)
{
	file->clearAttributes();
	
	unsigned i = 0;
	for(;i<attribs.length();i++) AddAttribute(attribs[i], file);
	
	file->setDirty();
	file->setWriteComponent(HesperisFile::WAttrib);
    bool fstat = file->save();
	if(!fstat) MGlobal::displayWarning(MString(" cannot save attrib to file ")+ file->fileName().c_str());
	file->close();
    AHelper::Info<unsigned>(" HesperisAttributeIO write n plugs", attribs.length() );
	return true;
}

bool HesperisAttributeIO::AddAttribute(const MPlug & attrib, HesperisFile * file)
{
    const std::string nodeName = H5PathNameTo(attrib.node());
	const std::string attrName = boost::str(boost::format("%1%|%2%") % nodeName % attrib.partialName().asChar());
	AHelper::Info<std::string>("HesperisAttributeIO add attrib ", attrName);
	
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
			AHelper::Info<std::string>("attr type not supported", attrName);
			break;
    }

	return true;
}

bool HesperisAttributeIO::ReadAttributeBundle(MObject &target)
{
    MGlobal::displayInfo("HesperisAttributeIO read attribute bundle");
    HWorld grpWorld;
    ReadAttributeBundle(&grpWorld, target);
    grpWorld.close();
    return true;
}

bool HesperisAttributeIO::ReadAttributeBundle(const ABundleAttribute * d,
                        const std::string & name,
                        MObject &target)
{
    std::cout<<"\n todo read attr bundle "<<name<<" n "<<d->shortName()<<" "<<d->longName()<<" "<<d->size();
    MObjectArray obs;
    LsChildren(obs, name, d->size(), target);
}

bool HesperisAttributeIO::ReadAttributeBundle(HBase * parent, MObject &target)
{
    std::vector<std::string > allGrps;
	std::vector<std::string > allAttrs;
	parent->lsTypedChild<HBase>(allGrps);
	std::vector<std::string>::const_iterator it = allGrps.begin();
	for(;it!=allGrps.end();++it) {
			
		HBase child(*it);
		if(child.hasNamedAttr(".bundle_num_typ")) {
            allAttrs.push_back(*it);
		}
		else {
			MObject otm;
			if( FindNamedChild(otm, child.lastName(), target) )
				ReadAttributeBundle(&child, otm);
		}
		child.close();
	}
	
	it = allAttrs.begin();
	for(;it!=allAttrs.end();++it) {

		HNumericBundle a(*it);
		ABundleAttribute wrap;
		if(a.load(&wrap)) {
		    ReadAttributeBundle(&wrap, a.parentName(), target);
		}
		a.close();
	}
	
	std::cout.flush();
	
	allAttrs.clear();
	allGrps.clear();
	
	return true;
}

bool HesperisAttributeIO::ReadAttributes(MObject &target)
{
	MGlobal::displayInfo("HesperisAttributeIO read attribute");
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
            //else {
            //    AHelper::Info<std::string>("HesperisAttributeIO cannot find child by name", child.fObjectPath );
            //}
		}
		child.close();
	}
	
	it = allAttrs.begin();
	for(;it!=allAttrs.end();++it) {
		HAttributeGroup a(*it);
		AAttributeWrap wrap;
		if(a.load(wrap)) {
			MObject attr;
			if(ReadAttribute(attr, wrap.attrib(), target)) {
				if(!ReadAnimation(&a, target, attr) ) {
					ConnectBaked(&a, wrap.attrib(), target, attr);
				}
			}
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

bool HesperisAttributeIO::BeginBakeAttribute(const std::string & attrName, ANumericAttribute *data)
{
	HBase grp(attrName);
	bool stat = true;
	HObject * d = CreateBake(&grp, data->numericType(), attrName, ".bake", stat );
	grp.close();
	if(stat) {
		AHelper::Info<std::string >("bake attr", attrName );
		MappedBakeData[attrName] = d;
	}
	else {
		AHelper::Info<std::string >("HesperisAttributeIO error cannot open group to bake attr", attrName );
	}
	return stat;
}

HObject * HesperisAttributeIO::CreateBake(HBase * grp, ANumericAttribute::NumericAttributeType typ,
										const std::string & attrName, const std::string & dataName,
										bool &stat)
{
    std::cout<<"\n HesperisAttributeIO create bake "<<grp->pathToObject()<<dataName
        <<" in file "<<HObject::FileIO.fileName();
	HObject * d = NULL;
	stat = false;
	switch(typ) {
		case ANumericAttribute::TByteNumeric:
			d = grp->createDataStorage<HOocArray<hdata::TChar, 1, 64> >(dataName, true, stat);
			break;
		case ANumericAttribute::TShortNumeric:
			d = grp->createDataStorage<HOocArray<hdata::TShort, 1, 64> >(dataName, true, stat);
			break;
		case ANumericAttribute::TIntNumeric:
			d = grp->createDataStorage<HOocArray<hdata::TInt, 1, 64> >(dataName, true, stat);
			break;
		case ANumericAttribute::TBooleanNumeric:
			d = grp->createDataStorage<HOocArray<hdata::TChar, 1, 64> >(dataName, true, stat);
			break;
		case ANumericAttribute::TFloatNumeric:
			d = grp->createDataStorage<HOocArray<hdata::TFloat, 1, 64> >(dataName, true, stat);
			break;
		case ANumericAttribute::TDoubleNumeric:
			d = grp->createDataStorage<HOocArray<hdata::TDouble, 1, 64> >(dataName, true, stat);
			break;
		default:
			break;
    }
	
	return d;
}

bool HesperisAttributeIO::EndBakeAttribute(const std::string & attrName, ANumericAttribute *data)
{
	if(MappedBakeData.find(attrName) == MappedBakeData.end() ) {
		std::cout<<"\n HesperisAttributeIO error end bake cannot find attr "<<attrName
        <<" in file "<<HObject::FileIO.fileName();
		return false;
	}
	HBase grp(attrName);
	
	FinishInsertDataValue(MappedBakeData[attrName], data->numericType() );
	
	grp.close();
	
	return true;
}

bool HesperisAttributeIO::BakeAttribute(const std::string & attrName, ANumericAttribute *data)
{
	if(MappedBakeData.find(attrName) == MappedBakeData.end() ) {
		std::cout<<"\n HesperisAttributeIO error cannot find attr "<<attrName
        <<" in file "<<HObject::FileIO.fileName();
		return false;
	}
	HBase grp(attrName);
	
	InsertDataValue(MappedBakeData[attrName], data);
	
	grp.close();
	return true;
}

bool HesperisAttributeIO::InsertDataValue(HObject * grp, ANumericAttribute *data)
{
	switch(data->numericType() ) {
		case ANumericAttribute::TByteNumeric:
			InsertValue<HOocArray<hdata::TChar, 1, 64>, char >(grp, static_cast<AByteNumericAttribute *>(data)->asChar() );
			break;
		case ANumericAttribute::TShortNumeric:
			InsertValue<HOocArray<hdata::TShort, 1, 64>, short >(grp, static_cast<AShortNumericAttribute *>(data)->value() );
			break;
		case ANumericAttribute::TIntNumeric:
			InsertValue<HOocArray<hdata::TInt, 1, 64>, int >(grp, static_cast<AIntNumericAttribute *>(data)->value() );
			break;
		case ANumericAttribute::TBooleanNumeric:
			InsertValue<HOocArray<hdata::TChar, 1, 64>, char >(grp, static_cast<ABooleanNumericAttribute *>(data)->asChar() );
			break;
		case ANumericAttribute::TFloatNumeric:
			InsertValue<HOocArray<hdata::TFloat, 1, 64>, float >(grp, static_cast<AFloatNumericAttribute *>(data)->value() );
			break;
		case ANumericAttribute::TDoubleNumeric:
			InsertValue<HOocArray<hdata::TDouble, 1, 64>, double >(grp, static_cast<ADoubleNumericAttribute *>(data)->value() );
			break;
		default:
			break;
    }
	return true;
} 

bool HesperisAttributeIO::FinishInsertDataValue(HObject * grp, ANumericAttribute::NumericAttributeType typ)
{
	switch(typ) {
		case ANumericAttribute::TByteNumeric:
			FinishInsertValue<HOocArray<hdata::TChar, 1, 64> >(grp);
			break;
		case ANumericAttribute::TShortNumeric:
			FinishInsertValue<HOocArray<hdata::TShort, 1, 64> >(grp);
			break;
		case ANumericAttribute::TIntNumeric:
			FinishInsertValue<HOocArray<hdata::TInt, 1, 64> >(grp);
			break;
		case ANumericAttribute::TBooleanNumeric:
			FinishInsertValue<HOocArray<hdata::TChar, 1, 64> >(grp);
			break;
		case ANumericAttribute::TFloatNumeric:
			FinishInsertValue<HOocArray<hdata::TFloat, 1, 64> >(grp);
			break;
		case ANumericAttribute::TDoubleNumeric:
			FinishInsertValue<HOocArray<hdata::TDouble, 1, 64> >(grp);
			break;
		default:
			break;
    }
	return true;
}

bool HesperisAttributeIO::BeginBakeEnum(const std::string & attrName, AEnumAttribute *data)
{
	HBase grp(attrName);
	bool stat = true;
	HObject * d = grp.createDataStorage<HOocArray<hdata::TShort, 1, 64> >(".bake", true, stat);
	grp.close();
	if(stat) {
		AHelper::Info<std::string >("bake enum attr", attrName );
		MappedBakeData[attrName] = d;
	}
	else {
		AHelper::Info<std::string >("HesperisAttributeIO error cannot bake enum attr", attrName );
	}
	return stat;
}

bool HesperisAttributeIO::EndBakeEnum(const std::string & attrName, AEnumAttribute *data)
{
	if(MappedBakeData.find(attrName) == MappedBakeData.end() ) {
		std::cout<<"\n HesperisAttributeIO error end bake cannot find enum attr "<<attrName;
		return false;
	}
	HBase grp(attrName);
	
	FinishInsertValue<HOocArray<hdata::TShort, 1, 64> >(MappedBakeData[attrName]);
	
	grp.close();
	
	return true;
}

bool HesperisAttributeIO::BakeEnum(const std::string & attrName, AEnumAttribute *data)
{
	if(MappedBakeData.find(attrName) == MappedBakeData.end() ) {
		std::cout<<"\n HesperisAttributeIO error cannot find enum attr "<<attrName;
		return false;
	}
	HBase grp(attrName);
	
	// AHelper::Info<short>("enum value ", data->asShort() );
	InsertValue<HOocArray<hdata::TShort, 1, 64>, short >(MappedBakeData[attrName], data->asShort() );
			
	grp.close();
	return true;
}

void HesperisAttributeIO::ClearBakeData()
{ MappedBakeData.clear(); }

bool HesperisAttributeIO::ConnectBaked(HBase * parent, AAttribute * data, MObject & entity, MObject & attr)
{
	if(!parent->hasNamedData(".bake")) return false;
	
	if(data->attrType() == AAttribute::aNumeric) {
		ANumericAttribute * numericData = static_cast<ANumericAttribute *>(data);
		HesperisAttribConnector::ConnectNumeric(parent->pathToObject(), numericData->numericType(), entity, attr);
	}
	else if(data->attrType() == AAttribute::aEnum) {
		HesperisAttribConnector::ConnectEnum(parent->pathToObject(), entity, attr);
	}		
	return true;
}

}
//:~
