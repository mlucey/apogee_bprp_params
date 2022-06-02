import numpy as np
from astropy.table import Table
import pandas as pd
import matplotlib.pyplot as p
from configparser import ConfigParser
from pathlib import Path
from os.path import join
from glob import glob
from gaiaxpy.calibrator.external_instrument_model import ExternalInstrumentModel
from gaiaxpy.config import config_path
from gaiaxpy.core.satellite import BANDS
from gaiaxpy.core import _get_spectra_type, _load_xpmerge_from_csv, \
                         _load_xpsampling_from_csv, _progress_tracker, \
                         _validate_arguments, _validate_wl_sampling, satellite
from gaiaxpy.input_reader import InputReader
from gaiaxpy.output import SampledSpectraData
from gaiaxpy.spectrum import _get_covariance_matrix, AbsoluteSampledSpectrum, \
                             SampledBasisFunctions, XpContinuousSpectrum

config_parser = ConfigParser()
config_parser.read(join(config_path, 'config.ini'))

def _create_merge(xp, sampling):
    """
    Create the weight information on the input sampling grid.
    Args:
        xp (str): Band (either BP or RP).
        sampling (ndarray): 1D array containing the sampling grid.
    Returns:
        dict: A dictionary containing a BP and an RP array with weights.
    """
    wl_high = satellite.BP_WL.high
    wl_low = satellite.RP_WL.low

    if xp == BANDS.bp:
        weight = np.array([1.0 if wl < wl_low else 0.0 if wl > wl_high else (
            1.0 - (wl - wl_low) / (wl_high - wl_low)) for wl in sampling])
    elif xp == BANDS.rp:
        weight = np.array([0.0 if wl < wl_low else 1.0 if wl > wl_high else (
            wl - wl_low) / (wl_high - wl_low) for wl in sampling])
    else:
        raise ValueError(f'Given band is {xp}, but should be either bp or rp.')
    return weight

def _generate_xp_matrices_and_merge(label, sampling, bp_model, rp_model):
    """
    Get xp_design_matrices and xp_merge.
    """
    def _get_file_for_xp(xp, key, bp_model=bp_model, rp_model=rp_model):
        file_name = config_parser.get(label, key)
        if xp == BANDS.bp:
            model = bp_model
        elif xp == BANDS.rp:
            model = rp_model
        return join(config_path, f"{file_name.replace('xp', xp).replace('model', model)}".format(key))

    xp_design_matrices = {}
    if sampling is None:
        xp_sampling_grid, xp_merge = _load_xpmerge_from_csv(label, bp_model=bp_model)
        xp_design_matrices = _load_xpsampling_from_csv(label, bp_model=bp_model)
        for xp in BANDS:
            xp_design_matrices[xp] = SampledBasisFunctions.from_design_matrix(
                xp_sampling_grid, xp_design_matrices[xp])
    else:
        xp_merge = {}
        for xp in BANDS:
            instr_model = ExternalInstrumentModel.from_config_csv(
                _get_file_for_xp(
                    xp, 'dispersion'), _get_file_for_xp(
                    xp, 'response'), _get_file_for_xp(
                    xp, 'bases'))
            xp_merge[xp] = _create_merge(xp, sampling)
            xp_design_matrices[xp] = SampledBasisFunctions.from_external_instrument_model(
                sampling, xp_merge[xp], instr_model)
    return xp_design_matrices, xp_merge

def convert():

    w = np.loadtxt('segue/Bp_4039.0_g3.101_f-0.406_c0.124_a0.44949996_n23.9.txt',skiprows=1,usecols=(0))


    xp_design_matrices, xp_merge = _generate_xp_matrices_and_merge('calibrator', w, 'v375wi', 'v142r')
    files = glob('segue/Bp_*')

    bp_coefs = []
    rp_coefs = []
    fs = []
    print(len(w))
    print(len(xp_merge['bp']))
    print(len(xp_merge['rp']))
    print(xp_design_matrices['bp']._get_design_matrix().shape)
    print(xp_design_matrices['rp']._get_design_matrix().shape)

    for i in range(len(files)):

        fbp = np.loadtxt(files[i],skiprows=1,usecols=(1))

        fbp = fbp
        #print(len(fbp))#
        #print(xp_design_matrices['bp'])
        bp_coef, r, rank, s  = np.linalg.lstsq(xp_design_matrices['bp']._get_design_matrix().T,fbp)
        bp_coefs.append(bp_coef)

        rp_f = 'segue/Rp_'+files[i][9:]

        frp = np.loadtxt(rp_f,skiprows=1,usecols=(1))

        frp = frp
        rp_coef, r, rank, s = np.linalg.lstsq(xp_design_matrices['rp']._get_design_matrix().T,frp)

        rp_coefs.append(rp_coef)
        fs.append(files[i][9:])

    t = Table()
    t['file']  = np.array(fs,dtype='str')
    t['bp_coefs'] = np.array(bp_coefs,dtype='float')
    t['rp_coefs'] = np.array(rp_coefs,dtype='float')
    t.write('segue_coef.fits',overwrite=True)


def plot_1():

    t = Table.read('segue_coef.fits')
    print(t['file'])

    fig, ax = p.subplots(2,2,figsize=(10,7),sharex='col',sharey=True)
    cindex = []
    ncindex = []

    for  i in range(len(t)):
        c = t['file'][i].split('_')
        c = float(c[3][1:])
        if c > 1:
            cindex.append(i)
            if len(cindex) == 1:
                break

    for i in range(len(t)):
        c = t['file'][i].split('_')
        c = float(c[3][1:])
        print(c)
        if c <1:
            ncindex.append(i)
            if len(ncindex) ==1:
                break

    cindex = np.array(cindex,dtype='int') ; ncindex = np.array(ncindex,dtype='int')
    print(cindex) ; print(ncindex)
    print(t['file'][ncindex])

    w, fbp, ebp= np.loadtxt('segue/Bp_'+str(np.array(t['file'][ncindex][0],dtype='str')),usecols=(0,1,2),skiprows=1,unpack=True)
    wr, frp, erp= np.loadtxt('segue/Rp_'+str(np.array(t['file'][ncindex][0],dtype='str')),usecols=(0,1,2),skiprows=1,unpack=True)

    xp_design_matrices, xp_merge = _generate_xp_matrices_and_merge('calibrator', w, 'v375wi', 'v142r')

    bp_m = xp_design_matrices['bp']._get_design_matrix()
    rp_m = xp_design_matrices['rp']._get_design_matrix()

    bp_f = t['bp_coefs'][ncindex].dot(bp_m)
    rp_f = t['rp_coefs'][ncindex].dot(rp_m)

    print(t['bp_coefs'][ncindex])


    bp_f_25 = t['bp_coefs'][ncindex][0][:25].dot(bp_m[:25][:])

    rp_f_25 = t['rp_coefs'][ncindex][0][:25].dot(rp_m[:25][:])

    bp_f_5 = t['bp_coefs'][ncindex][0][:5].dot(bp_m[:5][:])

    rp_f_5 = t['rp_coefs'][ncindex][0][:5].dot(rp_m[:5][:])


    bpc_f = t['bp_coefs'][cindex].dot(bp_m)
    rpc_f = t['rp_coefs'][cindex].dot(rp_m)


    bpc_f_25 = t['bp_coefs'][cindex][0][:25].dot(bp_m[:25][:])

    rpc_f_25 = t['rp_coefs'][cindex][0][:25].dot(rp_m[:25][:])

    bpc_f_5 = t['bp_coefs'][cindex][0][:5].dot(bp_m[:5][:])

    rpc_f_5 = t['rp_coefs'][cindex][0][:5].dot(rp_m[:5][:])

    w, fbpc, ebpc= np.loadtxt('segue/Bp_'+t['file'][cindex][0],usecols=(0,1,2),skiprows=1,unpack=True)
    wr, frpc, erpc= np.loadtxt('segue/Rp_'+t['file'][cindex][0],usecols=(0,1,2),skiprows=1,unpack=True)

    #fbpc = fbpc/10**22

    ax[0,0].plot(w,fbp,color='k')
    ax[0,0].plot(w,bp_f.flatten(),color='indianred')
    ax[0,0].plot(w,bp_f_25.flatten(),color='indianred',linestyle='--')
    #ax[0,0].plot(w,bp_f_5.flatten(),color='indianred',linestyle='dotted')

    ax[0,1].plot(w,frp,color='k')
    ax[0,1].plot(w,rp_f.flatten(),color='indianred')
    ax[0,1].plot(w,rp_f_25.flatten(),color='indianred',linestyle='--')
    #ax[0,1].plot(w,rp_f_5.flatten(),color='indianred',linestyle='dotted')

    ax[1,0].plot(w,fbpc,color='k')
    ax[1,0].plot(w,bpc_f.flatten(),color='indianred')
    ax[1,0].plot(w,bpc_f_25.flatten(),color='indianred',linestyle='--')
    #ax[1,0].plot(w,bpc_f_5.flatten(),color='indianred',linestyle='dotted')

    ax[1,1].plot(w,frpc,color='k')
    ax[1,1].plot(w,rpc_f.flatten(),color='indianred')
    ax[1,1].plot(w,rp_f_25.flatten(),color='indianred',linestyle='--')
    #ax[1,1].plot(w,rp_f_5.flatten(),color='indianred',linestyle='dotted')

    ax[0,0].set_ylim(0,1.5E-16)
    ax[1,0].set_ylim(0,1.5E-16)
    ax[0,1].set_ylim(0,1.5E-16)
    ax[1,1].set_ylim(0,1.5E-16)

    ax[0,0].set_xlim(300,700)
    ax[1,0].set_xlim(300,700)

    ax[0,1].set_xlim(600,1100)
    ax[1,1].set_xlim(600,1100)

    ax[0,0].set_ylabel('Flux')

    ax[1,0].set_ylabel('Flux')

    ax[1,0].set_xlabel('Wavelength')
    ax[1,1].set_xlabel('Wavelength')
