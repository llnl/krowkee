# Test executables

#
# MPI is required
#
find_package(MPI REQUIRED)

#
# This function adds a YGM test
#
function (add_ygm_krowkee_test test_name)
    set(test_source "${test_name}.cpp")
    set(test_exe "${test_name}")
    add_executable(${test_exe} ${test_source})
    target_link_libraries(${test_exe} PRIVATE krowkee)
    target_include_directories(
        ${test_exe}
        PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                $<INSTALL_INTERFACE:include>
    )
    target_include_directories(
        ${test_exe}
        PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
                $<INSTALL_INTERFACE:include>
    )
    if (${TEST_WITH_SLURM})
        add_test(${test_exe} "srun" "${CMAKE_CURRENT_BINARY_DIR}/${test_exe}")
    else ()
        message("mpiexec: ${MPIEXEC}")
        add_test(${test_exe} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} "4"
                 ${MPIEXEC_PREFLAGS} "${CMAKE_CURRENT_BINARY_DIR}/${test_exe}"
        )
    endif ()
endfunction ()

add_ygm_krowkee_test(power_iteration_test)
