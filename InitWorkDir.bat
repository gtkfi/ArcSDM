if exist ".\work\" rd /q /s ".\work"
mkdir .\work
xcopy /e .\data\* work
move .\work\ArcGisPro_MPM_DemoData.aprx .\work\ArcGisPro_MPM_DemoData_work.aprx 
move .\work\MPM_DemoData.mxd .\work\MPM_DemoData_work.mxd 
  