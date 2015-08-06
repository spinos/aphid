#include "H5FileIn.h"
#include <string>
#include <map>
class AField;
class H5FileOut : public H5FileIn {
public:
    H5FileOut();
    H5FileOut(const char * name);
    
    void addField(const std::string & fieldName,
                      AField * fld);
    
    AField * fieldByName(const std::string & fieldName);
    void writeFrame(int frame);
protected:

private:
    std::map<std::string, AField *> m_fields;
};
