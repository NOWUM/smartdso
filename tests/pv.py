from pvlib.pvsystem import PVSystem
import pandas as pd
from agents.utils import WeatherGenerator
from matplotlib import pyplot as plt
from pvlib.irradiance import get_total_irradiance

if __name__ == "__main__":

    model1 = PVSystem(module_parameters={'pdc0': 5.2, 'surface_tilt': 28, 'surface_azimuth': 159})
    model2 = PVSystem(module_parameters={'pdc0': 5.2, 'surface_tilt': 50, 'surface_azimuth': 70})

    weather = WeatherGenerator()
    w = weather.get_weather(pd.Timestamp(2022,5,8))
    rad1 = model1.get_irradiance(solar_zenith=w['zenith'], solar_azimuth=w['azimuth'],
                                 dni=w['dni'], ghi=w['ghi'], dhi=w['dhi'])
    rad2 = model2.get_irradiance(solar_zenith=w['zenith'], solar_azimuth=w['azimuth'],
                                 dni=w['dni'], ghi=w['ghi'], dhi=w['dhi'])

    rad3 = get_total_irradiance(solar_zenith=w['zenith'], solar_azimuth=w['azimuth'],
                                dni=w['dni'], ghi=w['ghi'], dhi=w['dhi'], surface_tilt=28,
                                surface_azimuth=159)
    rad4 = get_total_irradiance(solar_zenith=w['zenith'], solar_azimuth=w['azimuth'],
                                dni=w['dni'], ghi=w['ghi'], dhi=w['dhi'], surface_tilt=50,
                                surface_azimuth=70)

    plt.plot(rad1['poa_global'].values)
    plt.plot(rad2['poa_global'].values)
    plt.plot(rad3['poa_global'].values)
    plt.plot(rad4['poa_global'].values)

    plt.show()

