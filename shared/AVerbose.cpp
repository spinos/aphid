#include "AVerbose.h"
#include <iostream>

std::string AVerbose::verbosestr() const
{ return "AVerbose verbosestr()"; }

void AVerbose::verbose() const
{ std::cout<<verbosestr(); }
