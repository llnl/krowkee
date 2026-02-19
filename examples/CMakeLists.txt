# Test executables

#
# This function adds a Sequential test
#
function (add_krowkee_example example_name)
    set(example_source "${example_name}.cpp")
    set(example_exe "${example_name}")
    add_executable(${example_exe} ${example_source})
    target_link_libraries(${example_exe} PRIVATE krowkee)
    target_include_directories(
        ${example_exe}
        PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                $<INSTALL_INTERFACE:include>
    )
    add_test(${example_exe} "${CMAKE_CURRENT_BINARY_DIR}/${example_exe}")
endfunction ()

add_krowkee_example(cli)
add_krowkee_example(sparse_jlt)
add_krowkee_example(merge)
add_krowkee_example(power_iteration_iterative)
add_krowkee_example(power_iteration_streaming)
add_krowkee_example(verbose_templates)

if (KROWKEE_USE_YGM)
    add_subdirectory(ygm)
endif ()
