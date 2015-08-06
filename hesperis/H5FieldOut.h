#include "APlaybackFile.h"
#include <string>
#include <map>
class AField;
class H5FieldOut : public APlaybackFile {
public:
    H5FieldOut();
    H5FieldOut(const char * name);
    
    void addField(const std::string & fieldName,
                      AField * fld);
    
    AField * fieldByName(const std::string & fieldName);
    void writeFrame(int frame);
protected:

private:
    std::map<std::string, AField *> m_fields;
};
