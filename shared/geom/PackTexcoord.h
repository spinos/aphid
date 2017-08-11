/*
 *  PackTexcoord.h
 *
 *  pack multiple mesh texcoord into [0,1] region
 *  each mesh has its texcoord in [0,1]
 *  w/h ration and x or y fills [0,1]
 *
 */

#ifndef APH_PACKTEXCOORD_H
#define APH_PACKTEXCOORD_H

namespace aphid {
    
class ATriangleMesh;

class PackTexcoord {

public:
    PackTexcoord();
    
/// pack in rows when whratio < 1
/// shrink by 1/n_row or 1/n_col
    bool computeTexcoord(ATriangleMesh* msh[],
        const int& nmshs,
        const float& whratio);
    
protected:
    
private:
    bool packInRows(ATriangleMesh* msh[],
        const int& nmshs,
        const float& whratio);
    bool packInColumns(ATriangleMesh* msh[],
        const int& nmshs,
        const float& whratio);
    void shrinkAndMove(ATriangleMesh* msh,
        const float& shrinking,
        const float& xLoc,
        const float& yLoc);
    
};

}

#endif        //  #ifndef APH_PACKTEXCOORD_H

