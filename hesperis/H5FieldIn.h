#pragma once
#include "APlaybackFile.h"
#include <string>
#include <map>
class AField;
class H5FieldIn : public APlaybackFile {
public:
    H5FieldIn();
    H5FieldIn(const char * name);
	
	virtual bool doRead(const std::string & fileName);
    
    AField * fieldByName(const std::string & fieldName);
    AField * fieldByIndex(unsigned idx);

    virtual void addField(const std::string & fieldName,
                      AField * fld);
    
    unsigned numFields() const;
    
    bool readFrame();
protected:
    const std::map<std::string, AField *> * fields() const;
private:
    std::map<std::string, AField *> m_fields;
};
