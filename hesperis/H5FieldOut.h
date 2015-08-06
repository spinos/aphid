#include "H5FieldIn.h"

class H5FieldOut : public H5FieldIn {
public:
    H5FieldOut();
    H5FieldOut(const char * name);
    
    virtual void addField(const std::string & fieldName,
                      AField * fld);
    
    void writeFrame(int frame);
protected:

private:

};
