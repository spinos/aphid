#include "KnitPatch.h"
#include <Vector3F.h>

static float yarmU[14] = {0.f, .2f, .4f, .4f, .1f, .1f, .33f, .66f, .9f, .9f, .6f, .6f, .8f, 1.f };
static float yarmV[14] = {0.f, 0.f, .15f, .33f, .45f, .67f, .8f, .8f, .67f, .45f, .33f, .15f, 0.f, 0.f };
KnitPatch::KnitPatch() {}
KnitPatch::~KnitPatch() {}

unsigned KnitPatch::numYarnPoints() const
{
    return 14;
}

void KnitPatch::createYarn()
{
    const unsigned np = numYarnPoints();
    m_yarnP = new Vector3F[np];
    for(unsigned i = 0; i < np; i++) {
        m_yarnP[i] = interpolate(yarmU[i], yarmV[i]);
    }
}

Vector3F * KnitPatch::yarn()
{
    return m_yarnP;
}
