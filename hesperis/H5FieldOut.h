#pragma once
#include "H5FieldIn.h"
class AField;
class AdaptiveField;

class H5FieldOut : public H5FieldIn {
public:
    H5FieldOut();
    H5FieldOut(const char * name);
    
    virtual void addField(const std::string & fieldName,
                      AField * fld);
    
    void writeFrame(int frame);
protected:
    template<typename Th, typename Tf>
    void initField(const std::string & name,
                   Tf * field) {
        Th grp(name);
        grp.save(field);
        grp.close();
    }
private:
	void initFieldByType(const std::string & fieldName,  
							AField * fld);
};
