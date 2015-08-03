#include "AdaptiveGrid.h"

class AdaptiveField : public AdaptiveGrid
{
public:
    AdaptiveField(const BoundingBox & bound);
    virtual ~AdaptiveField();
    
    virtual void create(KdIntersection * tree, int maxLevel = 6);
protected:

private:
	
};
