// Define the interface to the word library.
namespace MY {
class Word {
	const char* _the_word;

public:
    Word(const char *w);

    const char *reverse() const;
};
};
