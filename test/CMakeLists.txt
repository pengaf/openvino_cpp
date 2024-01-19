set(ProjectName test)

set(AllFiles 
"test.cpp"
)

find_package(OpenVINO REQUIRED)
add_executable(${ProjectName} ${AllFiles})
target_link_libraries(${ProjectName} PRIVATE openvino::runtime)
target_link_libraries(${ProjectName} PRIVATE openvino::runtime::c)