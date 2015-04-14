#ifndef DYNAMICWORLDINTERFACE_H
#define DYNAMICWORLDINTERFACE_H
class BaseBuffer;
class GeoDrawer;
class CudaDynamicWorld;
class TetrahedronSystem;
class DynamicWorldInterface {
public:
    DynamicWorldInterface();
    virtual ~DynamicWorldInterface();
    
    virtual void create(CudaDynamicWorld * world);
    
    void draw(CudaDynamicWorld * world);
    void draw(CudaDynamicWorld * world, GeoDrawer * drawer);
protected:

private:
    void draw(TetrahedronSystem * tetra);
    void showOverlappingPairs(CudaDynamicWorld * world, GeoDrawer * drawer);
private:
    BaseBuffer * m_boxes;
    BaseBuffer * m_pairCache;
};
#endif        //  #ifndef DYNAMICWORLDINTERFACE_H

