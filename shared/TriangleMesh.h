#ifndef TRIANGLEMESH_H
#define TRIANGLEMESH_H

#include "BaseMesh.h"
class EasyModel;
class TriangleMesh : public BaseMesh {
public:
    TriangleMesh();
    virtual ~TriangleMesh();
    
    void copyOf(EasyModel * esm);
	virtual Matrix33F getTangentFrame(const unsigned& idx) const;
	
private:

};
#endif        //  #ifndef TRIANGLEMESH_H

