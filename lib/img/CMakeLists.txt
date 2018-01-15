file (GLOB img_SRC
    "../../shared/img/*.h"
    "../../shared/img/*.cpp"
)

add_library(aphidImg ${LIB_TYP}
${img_SRC}
)

SET (LIBRARY_OUTPUT_PATH ../)
