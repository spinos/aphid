#pragma once
class BezierPatch;
class BezierPatchHirarchy {
public:
    BezierPatchHirarchy();
    virtual ~BezierPatchHirarchy();
    void create(BezierPatch * parent, int maxLevel);
    
private:
    void recursiveCreate(BezierPatch * parent, int level, unsigned & current, unsigned & start);
    BezierPatch * m_elm;
    unsigned * m_childIdx;
    int m_maxLevel;
};
