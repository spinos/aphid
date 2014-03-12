// Define the interface to the word library.
namespace MY {
class Word {
	const char* the_word;

public:
    Word();

    char *reverse(const char *w) const;
};
};
