if exist ".\work\" rd /q /s ".\work"
mkdir .\work
xcopy /e .\data\* work
move .\work\arcsdm_orig.aprx .\work\ArcSDM_Work.aprx
move .\work\ArcSDM_Desktop_orig.mxd .\work\ArcSDM_Desktop_work.mxd
move .\work\ArcGisPro2_orig.aprx .\work\ArcGisPro2_work.aprx 
 