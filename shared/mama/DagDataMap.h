#ifndef MAMA_DAG_DATA_MAP_H
#define MAMA_DAG_DATA_MAP_H

#include <maya/MDagPathArray.h>
#include <map>
#include <string>

namespace aphid {
namespace mama {

template<typename T>
class DagDataMap {

    std::map<std::string, T * > m_data;
    
public:
    DagDataMap();
    virtual ~DagDataMap();
    
protected:
    bool hasNamedData(T * & dst, const std::string & name);
    void addData(const std::string & name, T * x );
    
private:
};

template<typename T>
DagDataMap<T>::DagDataMap()
{}

template<typename T>
DagDataMap<T>::~DagDataMap()
{
    typename std::map<std::string, T * >::iterator it = m_data.begin();
    for(;it!=m_data.end();++it) {
        delete it->second;
    }
    m_data.clear();
}

template<typename T>
bool DagDataMap<T>::hasNamedData(T * & dst, const std::string & name) {
    
    if(m_data.find(name) == m_data.end() ) {
        dst = NULL;
        return false;
    }
    dst = m_data[name];
    return true;
} 

template<typename T>
void DagDataMap<T>::addData(const std::string & name, T * x )
{
    m_data[name] = x;
}    

}
}
#endif
