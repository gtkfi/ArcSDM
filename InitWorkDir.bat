if exist ".\work\" rd /q /s ".\work"
mkdir .\work
xcopy /e .\data\* work
