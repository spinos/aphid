#include "H5FieldOut.h"
#include <HAdaptiveField.h>
#include <boost/format.hpp>

H5FieldOut::H5FieldOut() : H5FieldIn() {}
H5FieldOut::H5FieldOut(const char * name) : H5FieldIn(name) {}

void H5FieldOut::addField(const std::string & fieldName,
                      AField * fld)
{
    H5FieldIn::addField(fieldName, fld);
    useDocument();
    HField * grp = matchFieldType(fieldName, fld->fieldType());
    grp->save(fld);
    grp->close();
	delete grp;
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

HField * H5FieldOut::matchFieldType(const std::string & fieldName, 
									AField::FieldType typ) const
{
	HField * r;
	switch (typ) {
		case AField::FldAdaptive:
			r = new HAdaptiveField(fieldName);
			break;
		default:
			r = new HField(fieldName);
			break;
	}
	return r;
}
//:~
