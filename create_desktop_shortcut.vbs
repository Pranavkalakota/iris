' ============================================================
' Run this ONCE (double-click it) to drop an "IRIS" icon on the
' desktop that launches Launch_IRIS.bat. After this, demo day is
' just: double-click the IRIS icon on the desktop.
' ============================================================
Set oWS = WScript.CreateObject("WScript.Shell")
Set oFS = CreateObject("Scripting.FileSystemObject")
sScriptDir = oFS.GetParentFolderName(WScript.ScriptFullName)

sLinkFile = oWS.SpecialFolders("Desktop") & "\IRIS.lnk"
Set oLink = oWS.CreateShortcut(sLinkFile)
oLink.TargetPath = sScriptDir & "\Launch_IRIS.bat"
oLink.WorkingDirectory = sScriptDir
oLink.Description = "IRIS - wearable AI assistant"
oLink.Save

WScript.Echo "Done. An 'IRIS' icon was added to your desktop." & vbCrLf & _
             "From now on, just double-click that instead of any files in this folder."