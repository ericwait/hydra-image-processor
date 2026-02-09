$dll = "e:\programming\hydra-image-processor\src\Python\Hydra.pyd"
$result = & "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\dumpbin.exe" /DEPENDENTS $dll
Write-Output $result
