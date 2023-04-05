
from mlcrl.phasenet.zernike import Zernike
import numpy
from srxraylib.plot.gol import plot, plot_table
import matplotlib.pyplot as plt

def run_tubes(ITUBE=2, # 0=Mo, 1=Rh, 2= W
              VOLTAGE=30, # kV
              interpolate=0,
              do_plot=1,
              ):

    anode = ['Mo','Rh','W'][ITUBE]
    if VOLTAGE <=18 or VOLTAGE >=42:
        raise(Exception("Voltage (%f) must be 18<E<42" % VOLTAGE))

    #
    # script to make the calculations (created by XOPPY:xtubes)
    #
    import numpy
    import scipy.constants as codata
    from xoppylib.xoppy_run_binaries import xoppy_calc_xtubes

    out_file =  xoppy_calc_xtubes(
            ITUBE    = ITUBE,
            VOLTAGE = VOLTAGE,
            )
    data = numpy.loadtxt(out_file)

    # spectral_power = flux / 0.5e-3 * energy * codata.e # W/eV/mA/mm^2(@?m)
    # cumulated_power = spectral_power.cumsum() * numpy.abs(energy[1]-energy[0]) # W/mA/mm^2(@1m)

    # data to pass to power

    energy = data[:,0]
    flux = data[:,1] # for units, see discussion in 'help'


    if interpolate == 0: # crop
        energy_flag = numpy.ones_like(energy)
        energy_flag[energy < 1000] = 0
        energy_flag[energy > 40000] = 0
        # print(energy_flag)

        energy_i = energy[numpy.where(energy_flag>0)]
        flux_i = flux[numpy.where(energy_flag>0)]
    elif interpolate == 1: # interpolate
        energy_i = numpy.arange(1000, 40001, 500)
        flux_i = numpy.interp(energy_i, energy, flux)
        # energy = e
        # flux = f


    #
    # example plot
    #
    if do_plot:
        from srxraylib.plot.gol import plot

        plot(energy,flux,
             energy_i, flux_i,
            xtitle="Photon energy [eV]",ytitle="Fluence [photons/s/mm^2/0.5keV(bw)/mA]",
             title="xtubes Fluence; anode=%s,  V=%g kV, NPOINTS=%d" % (anode, VOLTAGE, energy.size),
            xlog=False,ylog=False,show=True, legend=["calculated", "cropped/interpolated"])
        # plot(energy,spectral_power,
        #     xtitle="Photon energy [eV]",ytitle="Spectral Power [W/eV/mA/mm^2(@?m) ]",title="xtube_w Spectral Power",
        #     xlog=False,ylog=False,show=False)
        # plot(energy,cumulated_power,
        #     xtitle="Photon energy [eV]",ytitle="Cumulated Power [W/mA/mm^2(@?m) ]",title="xtube_w Cumulated Power",
        #     xlog=False,ylog=False,show=True)

    #
    # end script
    #
    return energy_i, flux_i

def get_xrf(ITUBE=2, # 0=Mo, 1=Rh, 2= W
              VOLTAGE=30, # kV
              interpolate=0,
              do_plot=0,
              ):

    spectrum_e, spectrum_f = run_tubes(ITUBE=ITUBE, VOLTAGE=VOLTAGE, interpolate=interpolate, do_plot=do_plot)


    for i, energy in enumerate(spectrum_e):
        filename = "xrf_axo_%s_keV.csv" % repr(numpy.round(energy * 1e-3, 5))
        # print("filename: ", dir_path + filename)
        a = numpy.loadtxt(dir_path + filename, delimiter=',', skiprows=1)
        # print(a.shape)
        if do_plot > 1: plot(a[:,0], a[:,1], title="%s" % energy)
        if i == 0:
            xrf_e = a[:, 0]
            xrf_fcte = a[:, 1]
            xrf_f = spectrum_f[i] * a[:, 1]
        else:
            xrf_e = a[:, 0]
            xrf_fcte =+ a[:, 1]
            xrf_f += spectrum_f[i] * a[:, 1]

    return spectrum_e, spectrum_f,  xrf_e, xrf_f, xrf_fcte


if __name__ == "__main__":

    from srxraylib.plot.gol import plot
    #
    # retrieve and plot 1d (vertical) Zernike profiles
    #

    # Para cada energía utilicé 1e10 ph/s y un tiempo de colección del espectro de 100 ms.


    if True:
        dir_path = "/scisoft/data/srio/xrf_ml/"

        energies = numpy.arange(1.5, 41, 0.01)
        energies = numpy.arange(1.5, 41, 1)
        energies = numpy.arange(30.5, 41, 1)
        print(energies)



        for energy in energies:
            filename = "xrf_axo_%s_keV.csv" % repr(energy)
            print("filename: ", dir_path + filename)
            a = numpy.loadtxt(dir_path + filename, delimiter=',', skiprows=1)
            print(a.shape)
            plot(a[:,0], a[:,1], title="%s" % energy, ylog=1)

        # e, f = run_tubes(ITUBE=0, VOLTAGE=20, do_plot=1)
        # e, f = run_tubes(ITUBE=1, VOLTAGE=20, do_plot=1)
        # e, f = run_tubes(ITUBE=2, VOLTAGE=20, do_plot=1)


    if False:
        dir_path = "/tmp_14_days/reyesher/to_Manolo/XRF_spectra/"

        ITUBE = 2
        VOLTAGE = 40

        spectrum_e, spectrum_f, xrf_e, xrf_f, xrf_fcte = get_xrf(ITUBE=ITUBE, VOLTAGE=VOLTAGE, do_plot=1)

        plot(spectrum_e, spectrum_f,
             title="ITUBE=%d, VOLTAGE=%f" % (ITUBE, VOLTAGE), legend=["tube spectrum N=%d" % spectrum_e.size])

        plot(xrf_e, xrf_f,
             xrf_e, xrf_fcte / xrf_fcte.max() * xrf_f.max(),
             title="ITUBE=%d, VOLTAGE=%f, N=%d" % (ITUBE, VOLTAGE, xrf_e.size), legend=["with spectrum", "with cte spectrum"])


    if False: # version 01

        dir_path = "/tmp_14_days/reyesher/to_Manolo/XRF_spectra/"

        nsamples = 1000

        # X1 = numpy.random.randint(low=0, high=3, size=(nsamples,))
        X1 = numpy.ones(nsamples, dtype=int) * 2
        # print(X1)

        print(len(X1[numpy.where(X1 == 0)]))
        print(len(X1[numpy.where(X1 == 1)]))
        print(len(X1[numpy.where(X1 == 2)]))

        Y1 = numpy.random.rand(nsamples) * (42-18) + 18

        for i in range(nsamples):
            fileroot = "./data1/sampled_%05d" % (i)
            print(i, fileroot)

            spectrum_e, spectrum_f, xrf_e, xrf_f, xrf_fcte = get_xrf(ITUBE=X1[i], VOLTAGE=Y1[i], do_plot=0)

            numpy.savetxt(fileroot + "_spe.dat",
                          numpy.column_stack((spectrum_e, spectrum_f)) )

            numpy.savetxt(fileroot + "_xrf.dat",
                          numpy.column_stack((xrf_e, xrf_f, xrf_fcte)) )

            f = open(fileroot + ".txt", 'w')
            f.write("%d  %g\n" % (X1[i], Y1[i]))
            f.close()





