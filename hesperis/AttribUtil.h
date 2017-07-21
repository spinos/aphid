/*
 *  AttribUtil.h
 *  opium
 *
 *  Created by jian zhang on 10/17/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef OFL_HES_ATTRIB_UTIL_H
#define OFL_HES_ATTRIB_UTIL_H
#include "animUtil.h"
#include <maya/MPlugArray.h>

namespace aphid {
class AAttribute;
}

class AttribUtil : public AnimUtil {
public:
	AttribUtil();
	virtual ~AttribUtil();
	
	void save3(const MDagPathArray & entities);
	void saveBoundle(const MDagPathArray & entities,
	                const std::string & parentName);
	void saveUDANames(AnimIO& doc);
    void saveUserDefinedAttrib(AnimIO & doc, MObject & entity);
    void loadUserDefinedAttrib(AnimIO & doc, MObject & entity);
	void saveH5(const std::map<std::string, MDagPath > & entities);
/// write attrib at current frame, flag
/// :0 begin 
/// :1 reuse
/// :2 end
	void bakeH5(const std::map<std::string, MDagPath > & entities, int flag);
	void load3(const char * filename, MObject & target);
    
    static void ResetUserDefinedAttribFilter(const MString & arg);
	static AttribNameMap ListUserDefinedAttribs(const MString & nodeName, bool filtered = true);
	static bool IsFilterEmpty();
	
	virtual void dump(const char* filename, MDagPathArray &active_list);
	virtual void load(const char* filename, MObject &target = MObject::kNullObj);

protected:
    static AttribNameMap UserDefinedAttribFilter;
	
private:
	void scan(const MDagPath & entity);
	void saveAttrib(aphid::AAttribute::AttributeType t, AnimIO & doc, const MObject & entity, const MObject & attrib);
    void saveStringAttrib(AnimIO & doc, const MObject & node, const MObject & attrib);
    void saveNumericAttrib(AnimIO & doc, const MObject & node, const MObject & attrib, xmlNodePtr * parent = 0);
    void saveCompoundAttrib(AnimIO & doc, const MObject & node, const MObject & attrib);
    void saveEnumAttrib(AnimIO & doc, const MObject & node, const MObject & attrib);
    void loadAttrib(AnimIO & doc, MObject & entity, const AttribNameMap & existingAttribs);
    void loadStringAttrib(AnimIO & doc, MObject & entity, bool doCreate);
    void loadEnumAttrib(AnimIO & doc, MObject & entity, bool doCreate);
    void loadCompoundAttrib(AnimIO & doc, MObject & entity, bool doCreate);
    void loadNumericAttrib(AnimIO & doc, MObject & entity, bool doCreate);
    void setNumericAttrib(MFnDependencyNode & fdep, AnimIO & doc);
    MObject createNumericAttrib(AnimIO & doc);
	virtual void saveH5(const MPlug & attrib);
	void saveH5(const MObject & node, aphid::AAttribute * data);
    void bakeH5(const MPlug & attrib, int flag);
	void bakeNumeric(const MObject & entity, aphid::ANumericAttribute * data, int flag);
	void bakeEnum(const MObject & entity, aphid::AEnumAttribute * data, int flag);
	std::string fullAttrName(const std::string & nodeName, aphid::AAttribute * data) const;
/// write arrib through all frames
	void bakeAttrib(const char *filename, MDagPathArray &active_list);

private:
	MPlugArray m_dirtyPlugs;
};
#endif
