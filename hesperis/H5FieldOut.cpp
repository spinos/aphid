#include "H5FieldOut.h"
#include <AField.h>
#include <HField.h>
#include <boost/format.hpp>

H5FieldOut::H5FieldOut() : H5FieldIn() {}
H5FieldOut::H5FieldOut(const char * name) : H5FieldIn(name) {}

void H5FieldOut::addField(const std::string & fieldName,
                      AField * fld)
{
    H5FieldIn::addField(fieldName, fld);
    useDocument();
    HField grp(fieldName);
    grp.save(fld);
    grp.close();
}

void H5FieldOut::writeFrame(int frame)
{
    useDocument();
    const std::string sframe = boost::str(boost::format("%1%") % frame);
    std::map<std::string, AField *>::const_iterator it = fields()->begin();
    for(;it!=fields()->end(); ++it) {
        HField grp(it->first);
        grp.saveFrame(sframe, it->second);
        grp.close();
    }
}
//:~
