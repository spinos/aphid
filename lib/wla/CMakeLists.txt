file (GLOB wla_SRC
    "../../shared/wla/*.h"
    "../../shared/wla/*.cpp"
)

add_library(aphidWla ${LIB_TYP}
${wla_SRC}
)


SET (LIBRARY_OUTPUT_PATH ../)
