#ifndef DYNAMICWORLDINTERFACE_H
#define DYNAMICWORLDINTERFACE_H
class CudaDynamicWorld;
class TetrahedronSystem;
class DynamicWorldInterface {
public:
    DynamicWorldInterface();
    virtual ~DynamicWorldInterface();
    
    virtual void create(CudaDynamicWorld * world);
    
    void draw(CudaDynamicWorld * world);
protected:

private:
    void draw(TetrahedronSystem * tetra);
};
#endif        //  #ifndef DYNAMICWORLDINTERFACE_H

