#pragma once

#include <LODQuad.h>

class KnitPatch : public LODQuad {
public:
    KnitPatch();
    virtual ~KnitPatch();
    unsigned numYarnPoints() const;
    void createYarn();
    
    Vector3F * yarn();
private:
    Vector3F * m_yarnP;
};

