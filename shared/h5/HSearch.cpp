#include "HSearch.h"
#include <HBase.h>
#include <HTransform.h>
#include <foundation/SHelper.h>
#include <deque>

namespace aphid {
  
HSearch::HSearch()
{}

HSearch::~HSearch()
{}

HSearch::SearchResult HSearch::searchPathInFile(std::vector<std::string > & log,
                        const std::string & pathName,
                        const std::string & fileName)
{
    HDocument doc;
    if(!doc.open(fileName.c_str(), HDocument::oReadOnly) ) {
        return hFileNotOpen;
    }
    
    HObject::FileIO = doc;
    
    std::vector<std::string > hierachNames;
    SHelper::Split(pathName, hierachNames);
    const int maxIt = hierachNames.size();
    
    HBase w("/");
    
    HBase * head = &w;
    
    std::deque<HBase *> openedGrps;
    openedGrps.push_back(head);
    
    log.clear();
    SearchResult stat = hSuccess;
    int itr = 0;
    while(itr < maxIt) {
        const int nc = head->numChildren();
      
        bool found = false;
        const std::string tgtName = hierachNames[itr];
        for(int i=0;i<nc;++i) {
            if(!head->isChildGroup(i) ) {
                continue;   
            }
            
            std::string nodeName = head->childPath(i);
            SHelper::behead(nodeName, head->pathToObject());
            SHelper::behead(nodeName, "/");
            
            if(tgtName == nodeName) {
                std::cout<<"\n found "<<nodeName;
                log.push_back(nodeName);
                found = true;
                itr++;
                
                HBase * pchild = new HBase(head->childPath(i));
                
                openedGrps.push_front(pchild);
                
                head = pchild;
                
            }
        }
        
        if(!found) {
            std::cout<<"\n cannot find "<<tgtName;
            itr = maxIt + 1;
            stat = hPathNodeFound;
            
            log.push_back(":");
            for(int i=0;i<nc;++i) {
                if(!head->isChildGroup(i) ) {
                    continue;   
                }
                
                std::string nodeName = head->childPath(i);
                SHelper::behead(nodeName, head->pathToObject());
                SHelper::behead(nodeName, "/");
                
                log.push_back(nodeName);
            }
        }
    }
    
    const int nopend = openedGrps.size();
    for(int i=0;i<nopend;++i) {
        openedGrps[i]->close();
    }
    openedGrps.clear();
    
    if(!doc.close()) {
        return hFileNotClose;
    }
    
    return stat;
}

HSearch::SearchResult HSearch::listPathInFile(std::vector<std::string > & log,
                            const std::string & pathName,
                            const std::string & fileName)
{
    SearchResult stat = searchPathInFile(log, pathName, fileName);
    
    if(stat != hSuccess) {
        return stat;
    }
    
    HDocument doc;
    if(!doc.open(fileName.c_str(), HDocument::oReadOnly) ) {
        return hFileNotOpen;
    }
    
    HObject::FileIO = doc;

    HBase w(pathName);
    
    log.clear();
    const int nc = w.numChildren();
    for(int i=0;i<nc;++i) {
        if(!w.isChildGroup(i) ) {
            continue;   
        }
            
        std::string nodeName = w.childPath(i);
        SHelper::behead(nodeName, w.pathToObject());
        SHelper::behead(nodeName, "/");
        log.push_back(nodeName);   
        
    }
    
    w.close();

    if(!doc.close()) {
        return hFileNotClose;
    }
    
    return stat;
}

std::string HSearch::resultAsStr(SearchResult s) const
{
    switch(s) {
        case hFileNotOpen :
        return "file not open";
        case hFileNotClose :
        return "file not close";
    case hPathNodeFound :
        return "path not found";
    }
    return "success";
}

}
