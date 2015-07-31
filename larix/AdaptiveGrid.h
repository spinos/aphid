#include <CartesianGrid.h>
class KdIntersection;
class AdaptiveGrid : public CartesianGrid
{
public:
    AdaptiveGrid(const BoundingBox & bound);
    virtual ~AdaptiveGrid();
    
    void create(KdIntersection * tree);
protected:

private:

};
