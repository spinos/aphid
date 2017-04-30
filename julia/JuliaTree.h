#ifndef JULIATREE_H
#define JULIATREE_H

#include "Parameter.h"
namespace jul {

class JuliaTree {

public:
    JuliaTree(Parameter * param);
    virtual ~JuliaTree();
    
private:
    void buildTree(const std::string & name);
    void buildSphere(const std::string & name);
};

}
#endif        //  #ifndef JULIATREE_H

