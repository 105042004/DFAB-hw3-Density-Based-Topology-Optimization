if(TARGET igl::core)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG 66b3ef2253e765d0ce0db74cec91bd706e5ba176
)
FetchContent_MakeAvailable(libigl)
