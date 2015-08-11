#include "H5FieldIn.h"
#include <HAdaptiveField.h>
#include <AdaptiveField.h>
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
        AField * f = createTypedField(*it);
        std::cout<<"\n add field "<<*it;
        addField(*it, f);
    }
    
    verbose();
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
// todo keep fld typ
{ m_fields[fieldName] = fld; }

const std::map<std::string, AField *> * H5FieldIn::fields() const
{ return &m_fields; }

unsigned H5FieldIn::numFields() const
{ return m_fields.size(); }

bool H5FieldIn::readFrame()
{
    useDocument();
    const std::string sframe = currentFrameStr();
    
    std::map<std::string, AField *>::const_iterator it = m_fields.begin();
    for(;it!=m_fields.end(); ++it) {
        HField grp(it->first);
        grp.loadFrame(sframe, it->second);
        grp.close();
    }
    
    return true;
}

AField * H5FieldIn::createTypedField(const std::string & name)
{
	HBase g(name);
	int t = 0;
	g.readIntAttr(".fieldType", &t);
	g.close();
	
	if(t==AField::FldAdaptive)
		return createAdaptiveField(name);

	return createBaseField(name);
}

AField * H5FieldIn::createBaseField(const std::string & name)
{
	AField * f = new AField;
	HField gb(name);
	gb.load(f);
	gb.close();
	return f;
}

AdaptiveField * H5FieldIn::createAdaptiveField(const std::string & name)
{
	AdaptiveField * f = new AdaptiveField;
	HAdaptiveField gb(name);
	gb.load(f);
	gb.close();
	return f;
}

void H5FieldIn::verbose() const
{
    std::cout<<"\n h5 field file:"
    <<"\n n fields "<<numFields();
    
    std::map<std::string, AField *>::const_iterator it = m_fields.begin();
    for(;it!=m_fields.end();++it) {
        std::cout<<"\n field[\""<<it->first<<"\"]";
        ((AdaptiveField *)it->second)->verbose();
    }
    APlaybackFile::verbose();
}
//:~