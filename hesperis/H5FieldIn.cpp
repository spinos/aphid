#include "H5FieldIn.h"
#include <AField.h>
#include <HField.h>
#include <boost/format.hpp>

H5FieldIn::H5FieldIn() : APlaybackFile() {}
H5FieldIn::H5FieldIn(const char * name) : APlaybackFile(name) {}

bool H5FieldIn::doRead(const std::string & fileName)
{
	if(!HFile::doRead(fileName)) return false;
	
    APlaybackFile::readFrameRange();
    
    std::vector<std::string > names;
    HBase b("/");
    b.lsTypedChild<HField>(names);
    if(names.size() < 1) {
        std::cout<<"error: file has no field!";
        return false;
    }
    
    std::vector<std::string >::const_iterator it = names.begin();
    for(;it!=names.end();++it) {
        HField g(*it);
        AField * f = new AField;
        g.load(f);
        g.close();
        
        addField(*it, f);
    }
    
	return true;
}

AField * H5FieldIn::fieldByName(const std::string & fieldName)
{ 
    if(m_fields.find(fieldName) == m_fields.end()) return 0;
    return m_fields[fieldName]; 
}

AField * H5FieldIn::fieldByIndex(unsigned idx)
{
    unsigned i=0;
    
    std::map<std::string, AField *>::const_iterator it = m_fields.begin();
    for(;it!=m_fields.end();++it) {
        if(i==idx) return it->second;
        i++;
    }
    return 0;
}

void H5FieldIn::addField(const std::string & fieldName,
                      AField * fld)
{ m_fields[fieldName] = fld; }

const std::map<std::string, AField *> * H5FieldIn::fields() const
{ return &m_fields; }

unsigned H5FieldIn::numFields() const
{ return m_fields.size(); }
//:~
