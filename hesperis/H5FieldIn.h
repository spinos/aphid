#pragma once
#include "APlaybackFile.h"
#include <string>
#include <map>
#include <AField.h>
class AdaptiveField;

class H5FieldIn : public APlaybackFile {
public:
    H5FieldIn();
    H5FieldIn(const char * name);
	
	virtual bool doRead(const std::string & fileName);
    
    AField * fieldByName(const std::string & fieldName);
    AField * fieldByIndex(unsigned idx);
	
	AField::FieldType fieldTypeByName(const std::string & fieldName);
    AField::FieldType fieldTypeByIndex(unsigned idx);

    virtual void addField(const std::string & fieldName,
                      AField * fld);
    
    unsigned numFields() const;
    
    bool readFrame();
    
    virtual void verbose() const;
protected:
    typedef std::pair<AField::FieldType, AField *> TypeFieldPair;
    const std::map<std::string, TypeFieldPair> * fields() const;
private:
	AField * createTypedField(const std::string & name);
	AField * createBaseField(const std::string & name);
	AdaptiveField * createAdaptiveField(const std::string & name);
private:
	std::map<std::string, TypeFieldPair> m_fields;
};
