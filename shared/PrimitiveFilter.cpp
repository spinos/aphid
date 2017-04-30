#include "PrimitiveFilter.h"

namespace aphid {

PrimitiveFilter::PrimitiveFilter() {m_componentType = TFace;}

void PrimitiveFilter::setComponentFilterType(ComponentType ft)
{
    m_componentType = ft;
}

PrimitiveFilter::ComponentType PrimitiveFilter::getComponentFilterType() const
{
    return m_componentType;
}

}
