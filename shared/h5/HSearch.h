#ifndef APH_HSEARCH_H
#define APH_HSEARCH_H

#include <vector>
#include <string>

namespace aphid {

class HSearch {

public:
    enum SearchResult {
        hSuccess = 0,
        hFileNotOpen = 1,
        hFileNotClose = 2,
        hPathNodeFound = 3
    };
    
    HSearch();
    virtual ~HSearch();
    
/// search a specific path
/// if not found, store child names in last found group in log
    SearchResult searchPathInFile(std::vector<std::string > & log,
                            const std::string & pathName,
                            const std::string & fileName);
/// list child in a specific path
/// stored in log
    SearchResult listPathInFile(std::vector<std::string > & log,
                            const std::string & pathName,
                            const std::string & fileName);
/// list mesh in a specific path
/// stored in log
    SearchResult listMeshInFile(std::vector<std::string > & log,
                            const std::string & pathName,
                            const std::string & fileName);
    std::string resultAsStr(SearchResult s) const;
    
};

}
#endif        //  #ifndef APH_HSEARCH_H

