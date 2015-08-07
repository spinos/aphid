#pragma once
#include "H5FieldIn.h"
#include <AField.h>
#include <HField.h>
class H5FieldOut : public H5FieldIn {
public:
    H5FieldOut();
    H5FieldOut(const char * name);
    
    virtual void addField(const std::string & fieldName,
                      AField * fld);
    
    void writeFrame(int frame);
protected:

private:
	HField * matchFieldType(const std::string & fieldName,  
							AField::FieldType typ) const;
};
