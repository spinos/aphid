#ifndef ADENIUMWORLD_H
#define ADENIUMWORLD_H
class BvhTriangleSystem;
class BvhBuilder;
class TriangleSystem;
class AdeniumRender;
class AdeniumWorld {
public:
    AdeniumWorld();
    virtual ~AdeniumWorld();
    
    void setBvhBuilder(BvhBuilder * builder);
    void addTriangleSystem(BvhTriangleSystem * tri); 
    void initOnDevice();
    void draw();
    void resizeRenderArea(int w, int h);
private:
    void drawTriangle(TriangleSystem * tri);
private:
    BvhTriangleSystem * m_objects[32];
    AdeniumRender * m_image;
    unsigned m_numObjects;
};
#endif        //  #ifndef ADENIUMWORLD_H

