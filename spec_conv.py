import numpy as np
import numbers
from scipy.special import eval_hermite, gamma
from configparser import ConfigParser
from gaiaxpy.core.satellite import BANDS
from os import path
from .config import get_config, load_config
from gaiaxpy.config import config_path
from gaiaxpy.calibrator import ExternalInstrumentModel
from gaiaxpy.spectrum import SampledBasisFunctions, XpContinuousSpectrum, \
                             XpSampledSpectrum, _get_covariance_matri

config_parser = ConfigParser()
config_parser.read(path.join(config_path, 'config.ini'))
config_file = path.join(config_path, config_parser.get('converter', 'optimised_bases'))



def _hermite_function(n, x):
    if n == 0:
        return sqrt_4_pi * np.exp(-x ** 2. / 2.)
    elif n == 1:
        return sqrt_4_pi * np.exp(-x ** 2. / 2.) * np.sqrt(2.) * x
    c1 = np.sqrt(2. / n) * x
    c2 = -np.sqrt((n - 1) / n)
    return c1 * _hermite_function(n - 1, x) + c2 * _hermite_function(n - 2, x)

def _evaluate_hermite_function(n, x, w):
    if w > 0:
        return _hermite_function(n, x)
    else:
        return 0


def load_config_file():
    config_df = load_config(config_file)
    return config_df

def get_unique_basis_ids(parsed_input_data):
    """
    Get the IDs of the unique basis required to sample all spectra in the input files.
    Args:
        parsed_input_data (DataFrame): Pandas DataFrame populated with the content
            of the file containing the mean spectra in continuous representation.
    Returns:
        set: A set containing all the required unique basis function IDs.
    """
    # Keep only non NaN values (in Python, nan != nan)
    def remove_nans(_set):
        return {int(element) for element in _set if element == element}

    set_bp = set([basis for basis in parsed_input_data[f'{BANDS.bp}_basis_function_id'] if isinstance(basis, numbers.Number)])
    set_rp = set([basis for basis in parsed_input_data[f'{BANDS.rp}_basis_function_id'] if isinstance(basis, numbers.Number)])
    return remove_nans(set_bp).union(remove_nans(set_rp))


def get_design_matrices(unique_bases_ids, sampling, config_df):
    """
    Get the design matrices corresponding to the input bases.
    Args:
        unique_bases_ids (set): A set containing the basis function IDs
            for which the design matrix is required.
        sampling (ndarray): 1D array containing the sampling grid.
        config_df (DataFrame): A DataFrame containing the configuration for
            all sets of basis functions.
    Returns:
        list: a list of the design matrices for the input list of bases.
    """
    design_matrices = {}
    for id in unique_bases_ids:
        design_matrices.update({id: SampledBasisFunctions.from_config(
            sampling, get_config(config_df, id))})
    return design_matrices

def populate_design_matrix(sampling_grid, config):
    """
    Create a design matrix given the internal calibration bases and a user-defined
    sampling.
    Args:
        sampling_grid (ndarray): 1D array of positions where the bases need to
                be evaluated.
        config (DataFrame): The configuration of the set of bases
                loaded into a DataFrame.
    Returns:
        ndarray: The resulting design matrix.
    """
    #convert wavelength grid to pixel sampling
    n_samples = len(sampling_grid)
    scale = (config['normalizedRange'].iloc(0)[0][1] - config['normalizedRange'].iloc(0)
             [0][0]) / (config['range'].iloc(0)[0][1] - config['range'].iloc(0)[0][0])
    offset = config['normalizedRange'].iloc(0)[0][0] - config['range'].iloc(0)[0][0] * scale
    rescaled_pwl = (sampling_grid * scale) + offset

    def psi(n, x): return 1.0 / np.sqrt((2 ** n * gamma(n + 1) *
                                         np.sqrt(np.pi))) * np.exp(-x ** 2 / 2.0) * eval_hermite(n, x)

    bases_transformation = config['transformationMatrix'].iloc(0)[0].reshape(
        int(config['dimension']), int(config['transformedSetDimension']))

    design_matrix = np.array([psi(n_h, pos) for pos in rescaled_pwl for n_h in np.arange(
        int(config['dimension']))]).reshape(n_samples, int(config['dimension']))

    return bases_transformation.dot(design_matrix.T)


def from_external_instrument_model(sampling, weights, external_instrument_model):
    """
    Instantiate an object starting from a sampling grid, an array of weights and the
    external calibration instrument model.
    Args:
        sampling (ndarray): 1D array of positions where the bases need to
            be evaluated.
        weights (ndarray): 1D array containing the weights to be applied at each
            element in the sampling grid. These are simply used to define where
            in the sampling grid some contribution is expected. Where the weight is
            0, the bases will not be evaluated.
        external_instrument_model (obj): external calibration instrument model.
            This object contains information on the dispersion, response and
            inverse bases.
    Returns:
        object: An instance of this class.
    """
    n_samples = len(sampling)
    scale = (external_instrument_model.bases['normRangeMax'][0] - external_instrument_model.bases['normRangeMin'][0]) / (
        external_instrument_model.bases['pwlRangeMax'][0] - external_instrument_model.bases['pwlRangeMin'][0])
    offset = external_instrument_model.bases['normRangeMin'][0] - \
        external_instrument_model.bases['pwlRangeMin'][0] * scale

    sampling_pwl = external_instrument_model._wl_to_pwl(sampling)
    rescaled_pwl = (sampling_pwl * scale) + offset

    bases_transformation = external_instrument_model.bases['transformationMatrix'][0]
    evaluated_hermite_bases = np.array(
        [
            _evaluate_hermite_function(
                n_h, pos, weight) for pos, weight in zip(
                rescaled_pwl, weights) for n_h in np.arange(
                int(
                    external_instrument_model.bases['nInverseBasesCoefficients'][0]))]) .reshape(
                        n_samples, int(
                            external_instrument_model.bases['nInverseBasesCoefficients'][0]))
    _design_matrix = external_instrument_model.bases['inverseBasesCoefficients'][0]\
        .dot(evaluated_hermite_bases.T)

    transformed_design_matrix = bases_transformation.dot(_design_matrix)

    hc = 1.e9 * nature.C * nature.PLANCK

    def compute_norm(wl):
        r = external_instrument_model._get_response(wl)
        if r > 0:
            return hc / (satellite.TELESCOPE_PUPIL_AREA * r * wl)
        else:
            return 0.

    norm = np.array([compute_norm(wl) for wl in sampling])

    design_matrix = np.zeros(_design_matrix.shape)
    for i in np.arange(external_instrument_model.bases['nBases'][0]):
        design_matrix[i] = transformed_design_matrix[i] * norm

    return design_matrix
