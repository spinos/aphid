file (GLOB pbd_SRC
    "../../shared/pbd/*.h"
    "../../shared/pbd/*.cpp"
)

add_library(aphidPbd ${LIB_TYP}
${pbd_SRC}
)

SET (LIBRARY_OUTPUT_PATH ../)
