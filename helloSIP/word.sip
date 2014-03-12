// Define the SIP wrapper to the word library.

%Module (name=word, language="C++")
namespace MY {
class Word {

%TypeHeaderCode
#include <word.h>
%End

public:
    Word();

    char *reverse(const char * w) const;
};
};
