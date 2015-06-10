#ifndef ADENIUMINTERFACE_H
#define ADENIUMINTERFACE_H
#include <string>
class AdeniumWorld;
class AdeniumInterface {
public:
    AdeniumInterface();
    virtual ~AdeniumInterface();
    void create(AdeniumWorld * world);
    static std::string FileName;
protected:
    bool readTriangleMeshFromFile(AdeniumWorld * world);
};
#endif        //  #ifndef ADENIUMINTERFACE_H

