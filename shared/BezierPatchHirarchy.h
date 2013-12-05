#pragma once
class BezierPatch;
class BezierPatchHirarchy {
public:
    BezierPatchHirarchy();
    virtual ~BezierPatchHirarchy();
	void cleanup();
    void create(BezierPatch * parent);
    char isEmpty() const;
    char endLevel(int level) const;
    
    BezierPatch * patch(unsigned idx) const;
    BezierPatch * childPatch(unsigned idx) const;
    unsigned childStart(unsigned idx) const;
private:
    void recursiveCreate(BezierPatch * parent, short level, unsigned & current, unsigned & start);
    BezierPatch * m_elm;
    unsigned * m_childIdx;
};
