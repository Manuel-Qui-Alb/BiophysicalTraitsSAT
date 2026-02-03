from specdal.spectrum import Spectrum
import matplotlib.pyplot as plt
import os

folder = os.path.normpath(rf'C:\Users\mqalborn\Box\DATA_CUBBIES\Manuel\BLS\FieldSpec\280126')
asd_path = folder + "/#8LS#11100000.asd"

s = Spectrum(filepath=asd_path)
# s.read(measure_type="pct_reflect")
# reflectance in percent (commonly) # reads one file (you can also read a folder)

wl = s.measurement.index.values
refl = s.measurement.values

plt.plot(wl, refl)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance (%)")
plt.ylim(0, 1)
plt.title(asd_path.split("/")[-1])
plt.grid(True, alpha=0.3)
plt.show()
