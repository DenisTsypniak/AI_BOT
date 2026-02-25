@echo off
echo Installing requirements for DB Viewer...
python -m pip install -r "db_viewer\requirements.txt"
if errorlevel 1 goto :fail

echo.
echo Running template smoke check...
python -m db_viewer.check_templates
if errorlevel 1 goto :fail

echo.
echo Starting DB Viewer Dashboard...
echo Dashboard will be available at: http://127.0.0.1:8000
echo.
python -m db_viewer.app
goto :eof

:fail
echo.
echo DB Viewer startup aborted due to previous error.
exit /b 1
