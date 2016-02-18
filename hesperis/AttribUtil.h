/*
 *  AttribUtil.h
 *  opium
 *
 *  Created by jian zhang on 10/17/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "animUtil.h"
#include <maya/MPlugArray.h>
class AAttribute;
class AttribUtil : public AnimUtil {
public:
	AttribUtil();
	virtual ~AttribUtil();
	
	void save3(const MDagPathArray & entities);
	void saveUDA(AnimIO& doc);
    void saveUserDefinedAttrib(AnimIO & doc, MObject & entity);
    void loadUserDefinedAttrib(AnimIO & doc, MObject & entity);
	void saveH5(const std::map<std::string, MDagPath > & entities);
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
	void saveAttrib(AAttribute::AttributeType t, AnimIO & doc, const MObject & entity, const MObject & attrib);
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
	void saveH5(const MObject & node, AAttribute * data);
    void bakeH5(const MPlug & attrib, int flag);
	void bakeNumeric(const MObject & entity, ANumericAttribute * data, int flag);
	void bakeEnum(const MObject & entity, AEnumAttribute * data, int flag);
	std::string fullAttrName(const std::string & nodeName, AAttribute * data) const;
	void bakeAttrib(const char *filename, MDagPathArray &active_list);

private:
	MPlugArray m_dirtyPlugs;
};