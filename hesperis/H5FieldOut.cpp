#include "H5FieldOut.h"
#include <AField.h>
#include <HField.h>
#include <boost/format.hpp>

H5FieldOut::H5FieldOut() : APlaybackFile() {}
H5FieldOut::H5FieldOut(const char * name) : APlaybackFile(name) {}

void H5FieldOut::addField(const std::string & fieldName,
                      AField * fld)
{
    m_fields[fieldName] = fld;
    
    useDocument();
    HField grp(fieldName);
    grp.save(fld);
    grp.close();
}

AField * H5FieldOut::fieldByName(const std::string & fieldName)
{ return m_fields[fieldName]; }

void H5FieldOut::writeFrame(int frame)
{
    useDocument();
    const std::string sframe = boost::str(boost::format("%1%") % frame);
    std::map<std::string, AField *>::const_iterator it = m_fields.begin();
    for(;it!=m_fields.end(); ++it) {
        HField grp(it->first);
        grp.saveFrame(sframe, it->second);
        grp.close();
    }
}
//:~
