#include "StripeMap.h"
#include <AStripedModel.h>
#include <BaseBuffer.h>
StripeMap::StripeMap() : SplineMap1D(1.f, 0.5f)
{
    m_stripeIndices = new BaseBuffer;
}
StripeMap::~StripeMap() 
{
    delete m_stripeIndices;
}

void StripeMap::create(AStripedModel * mdl)
{
    m_numStripes = mdl->numStripes();
    m_stripeIndices->create(m_numStripes * 4);
    m_stripeIndices->copyFrom(mdl->indexDrifts(), m_numStripes * 4);
    //unsigned i = 0;
    //for(;i<m_numStripes;i++) std::cout<<" "<<stripeBegin(i); 
}

unsigned * StripeMap::stripeBegins() const
{ return (unsigned *)m_stripeIndices->data();} 

unsigned StripeMap::stripeBegin(unsigned i) const
{ return stripeBegins()[i]; }

void StripeMap::setLastIndex(unsigned x)
{ m_lastIndex = x; }

void StripeMap::computeTetrahedronInStripe(float * dst, unsigned n)
{
    unsigned i=0;
    unsigned tetv;
    unsigned iStripe = 0;
    unsigned firstInStripe = stripeBegin(iStripe);
    unsigned lastInStripe = stripeBegin(iStripe+1);
    float alpha;
    for(;i<n;i++) {
        tetv = i*4;
        alpha = (float)tetv - (float)firstInStripe;
        alpha /= (float)lastInStripe - (float)firstInStripe;
        dst[i] = interpolate(alpha);

        if(tetv == lastInStripe) {
            iStripe++;
            firstInStripe = stripeBegin(iStripe);
            if(iStripe < m_numStripes-1)
                lastInStripe = stripeBegin(iStripe+1);
            else
                lastInStripe = m_lastIndex;
        }
    }
}
