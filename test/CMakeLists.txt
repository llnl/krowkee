# Test executables

#
# This function adds a Sequential test
#
function (add_seq_krowkee_test test_name)
    set(test_source "${test_name}.cpp")
    set(test_exe "${test_name}")
    add_executable(${test_exe} ${test_source})
    target_link_libraries(${test_exe} PRIVATE krowkee)
    target_include_directories(
        ${test_exe}
        PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                $<INSTALL_INTERFACE:include>
    )
    add_test(${test_exe} "${CMAKE_CURRENT_BINARY_DIR}/${test_exe}")
endfunction ()

add_seq_krowkee_test(hash_test)
add_seq_krowkee_test(countsketch_hash_test)
add_seq_krowkee_test(test_fwht_utils)
add_seq_krowkee_test(compacting_map_test)
add_seq_krowkee_test(linearsketch_test)
add_seq_krowkee_test(two_sided_test)
