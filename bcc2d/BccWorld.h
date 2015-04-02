#ifndef BCCWORLD_H
#define BCCWORLD_H
class BezierCurve;
class LineDrawer;
class BccWorld {
public:
    BccWorld(LineDrawer * drawer);
    virtual ~BccWorld();
    
    void draw();
private:
    LineDrawer * m_drawer;
    BezierCurve * m_curve;
    BezierCurve * m_segmentCurve;
};

#endif        //  #ifndef BCCWORLD_H

