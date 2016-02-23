#pragma once

namespace aphid {

class PrimitiveFilter {
public:
    enum ComponentType {
		TFace,
		TEdge,
		TVertex
	};
	
    PrimitiveFilter();
    
    void setComponentFilterType(ComponentType ft);
    ComponentType getComponentFilterType() const;
    
private:
    ComponentType m_componentType;
};

}

