if exist ".\work\" rd /q /s ".\work"
mkdir .\work
xcopy /e .\data\* work
move .\work\arcsdm_orig.aprx .\work\ArcSDM_Work.aprx
 