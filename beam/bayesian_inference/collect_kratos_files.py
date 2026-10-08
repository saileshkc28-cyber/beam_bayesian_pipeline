import os, shutil

ROOT = r"D:\KratosFiles\Kratos"
FOLDERS = [
    r"bin\Release\KratosMultiphysics\SystemIdentificationApplication",
    r"bin\Release\KratosMultiphysics\OptimizationApplication",
    r"applications\SystemIdentificationApplication\custom_responses",
    r"applications\SystemIdentificationApplication\custom_sensors",
]
OUT = "kratos_si_files"
skip = shutil.ignore_patterns("__pycache__", "*.pyd", "*.dll", "*.lib", "*.pdb")

if os.path.exists(OUT):
    shutil.rmtree(OUT)
for rel in FOLDERS:
    src = os.path.join(ROOT, rel)
    if os.path.isdir(src):
        shutil.copytree(src, os.path.join(OUT, rel), ignore=skip)
        print("copied   ", rel)
    else:
        print("NOT FOUND", src)

shutil.make_archive(OUT, "zip", OUT)
print("upload this file:", os.path.abspath(OUT + ".zip"))
