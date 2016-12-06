#include "AVerbose.h"
#include <iostream>
namespace aphid {

std::string AVerbose::verbosestr() const
{ return "AVerbose verbosestr()"; }

void AVerbose::verbose() const
{ std::cout<<verbosestr(); }

}
