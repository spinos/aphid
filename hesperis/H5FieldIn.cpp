#include "H5FieldIn.h"
#include <AField.h>
#include <HField.h>
#include <boost/format.hpp>

H5FieldIn::H5FieldIn() : APlaybackFile() {}
H5FieldIn::H5FieldIn(const char * name) : APlaybackFile(name) {}

AField * H5FieldIn::fieldByName(const std::string & fieldName)
{ return m_fields[fieldName]; }

void H5FieldIn::addField(const std::string & fieldName,
                      AField * fld)
{ m_fields[fieldName] = fld; }

const std::map<std::string, AField *> * H5FieldIn::fields() const
{ return &m_fields; }
