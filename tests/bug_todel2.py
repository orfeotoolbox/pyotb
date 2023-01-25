import otbApplication
img_pth = "/vsicurl/https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb/-/raw/develop/Data/Input/SP67_FR_subset_1.tif?inline=false"
app = otbApplication.Registry.CreateApplication("BandMath")
app.SetParameterStringList("il", [img_pth])
app.SetParameterString("exp", "im1b1")
app.Execute()
print(app.GetImageSize("out"))  # WORKS
app.SetParameterString("out", "/tmp/toto.tif")
app.WriteOutput()
print(app.GetImageSize("out"))  # ERROR
