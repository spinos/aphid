#ifndef HES_SCENE_H
#define HES_SCENE_H

#include <string>

namespace aphid {

class HesScene {

public:
    HesScene();
    virtual ~HesScene();
    
    bool load(const std::string& fileName);
    
protected:

private:
};

}
#endif
