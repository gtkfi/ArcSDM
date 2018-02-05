if exist ".\work\" rd /q /s ".\work"
mkdir .\work
xcopy /e .\data\* work
move .\work\DemoData_version_files\MPM_DemoFiles.mxd .\work\DemoData_version_files\MPM_DemoFiles_work.mxd 
move .\work\DemoData_version_gdb\MPM_DemoGdb.mxd .\work\DemoData_version_gdb\MPM_DemoGdb_work.mxd 
  